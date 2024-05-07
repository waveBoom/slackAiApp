import os
import logging
import hashlib
import random
import uuid
import tiktoken
from pathlib import Path
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason, \
    SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from llama_index.core import Document, ServiceContext, StorageContext, VectorStoreIndex, Settings, \
    SimpleDirectoryReader, load_index_from_storage, SummaryIndex
from llama_index.core.indices import SimpleKeywordTableIndex
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.legacy.readers import RssReader
from app.fetch_web_post import get_urls, get_youtube_transcript, scrape_website, scrape_website_by_phantomjscloud
from app.prompt import get_prompt_template
from app.util import get_language_code, get_youtube_video_id, md5
from openai import OpenAI
from openai.types.audio import Transcription

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
model_name = "gpt-3.5-turbo"

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# openai.api_key = OPENAI_API_KEY
SPEECH_KEY = os.environ.get('SPEECH_KEY')
SPEECH_REGION = os.environ.get('SPEECH_REGION')

index_cache_web_dir = Path('/tmp/myGPTReader/cache_web/')
index_cache_file_dir = Path('/data/myGPTReader')
index_cache_voice_dir = Path('/tmp/myGPTReader/voice/')

if not index_cache_web_dir.is_dir():
    index_cache_web_dir.mkdir(parents=True, exist_ok=True)

if not index_cache_voice_dir.is_dir():
    index_cache_voice_dir.mkdir(parents=True, exist_ok=True)

if not index_cache_file_dir.is_dir():
    index_cache_file_dir.mkdir(parents=True, exist_ok=True)


from llama_index.llms.openai import OpenAI as llmOpenAi
llm = llmOpenAi(temperature=0, model=model_name)

service_context = ServiceContext.from_defaults(llm=llm)

web_storage_context = StorageContext.from_defaults()
file_storage_context = StorageContext.from_defaults()

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model(model_name).encode
)
Settings.callback_manager = CallbackManager([token_counter])

def get_unique_md5(urls):
    urls_str = ''.join(sorted(urls))
    hashed_str = hashlib.md5(urls_str.encode('utf-8')).hexdigest()
    return hashed_str


def format_dialog_messages(messages):
    return "\n".join(messages)


def format_dialog_messages_to_gpt(messages):
    formatted_messages = []
    for message in messages:
        # 忽略空字符串
        if not message.strip():
            continue
        if message.startswith('User:'):
            role = 'user'
            content = message[len('User:'):].strip()  # 移除前缀和前后的空白字符
        elif message.startswith('chatGPT:'):
            role = 'assistant'
            content = message[len('chatGPT:'):].strip()  # 移除前缀和前后的空白字符
        else:
            # 如果不是以'User:'或'Chatgpt:'开头，跳过处理
            continue
        formatted_messages.append({"role": role, "content": content})
    return formatted_messages


def get_document_from_youtube_id(video_id):
    if video_id is None:
        return None
    transcript = get_youtube_transcript(video_id)
    if transcript is None:
        return None
    return Document(text=transcript)


def remove_prompt_from_text(text):
    return text.replace('chatGPT:', '').replace('User:', '').replace("Assistant:", '').strip()


def get_documents_from_urls(urls):
    documents = []
    for url in urls['page_urls']:
        document = Document(text=scrape_website(url))
        documents.append(document)
    if len(urls['rss_urls']) > 0:
        rss_documents = RssReader().load_data(urls['rss_urls'])
        documents = documents + rss_documents
    if len(urls['phantomjscloud_urls']) > 0:
        for url in urls['phantomjscloud_urls']:
            document = Document(text=scrape_website_by_phantomjscloud(url))
            documents.append(document)
    if len(urls['youtube_urls']) > 0:
        for url in urls['youtube_urls']:
            video_id = get_youtube_video_id(url)
            document = get_document_from_youtube_id(video_id)
            if (document is not None):
                documents.append(document)
            else:
                documents.append(Document(text=f"Can't get transcript from youtube video: {url}"))
    return documents


def get_index_from_web_cache(name):
    try:
        index = load_index_from_storage(web_storage_context, index_id=name)
    except Exception as e:
        logging.error(e)
        return None
    return index


def get_index_from_file_cache(name):
    try:
        index = load_index_from_storage(file_storage_context, index_id=name)
    except Exception as e:
        logging.error(e)
        return None
    return index


def get_index_name_from_file(file: str):
    file_md5_with_extension = str(Path(file).relative_to(index_cache_file_dir).name)
    file_md5 = file_md5_with_extension.split('.')[0]
    return file_md5


def get_answer_from_chatGPT(messages):
    dialog_messages = format_dialog_messages_to_gpt(messages)
    logging.info('=====> Use chatGPT to answer!')
    logging.info(dialog_messages)
    completion = client.chat.completions.create(
        messages=dialog_messages,
        model=model_name,
    )
    logging.info(completion.usage)
    total_tokens = completion.usage.total_tokens
    res = remove_prompt_from_text(str(completion.choices[0].message.content))

    return res, total_tokens, None



def get_answer_from_llama_web(messages, urls):
    dialog_messages = format_dialog_messages(messages)
    lang_code = get_language_code(remove_prompt_from_text(messages[-1]))
    combained_urls = get_urls(urls)
    logging.info(combained_urls)
    index_file_name = get_unique_md5(urls)
    index = get_index_from_web_cache(index_file_name)
    if index is None:
        logging.info(f"=====> Build index from web!")
        documents = get_documents_from_urls(combained_urls)
        logging.info(documents)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.set_index_id(index_file_name)
        index.storage_context.persist()
        logging.info(
            f"=====> Save index to disk path: {index_cache_web_dir / index_file_name}")
    prompt = get_prompt_template(lang_code)
    logging.info('=====> Use llama web with chatGPT to answer!')
    logging.info('=====> dialog_messages')
    logging.info(dialog_messages)
    logging.info('=====> text_qa_template')
    answer = index.as_query_engine(text_qa_template=prompt, similarity_top_k=5).query(dialog_messages)
    answer.response = remove_prompt_from_text(answer.response)
    total_embedding_token_count = token_counter.total_embedding_token_count
    total_llm_token_count = token_counter.total_llm_token_count

    token_counter.reset_counts()
    return answer, total_llm_token_count, total_embedding_token_count


def get_answer_from_llama_file(messages, file):
    dialog_messages = format_dialog_messages(messages)
    lang_code = get_language_code(remove_prompt_from_text(messages[-1]))
    index_name = get_index_name_from_file(file)
    index = get_index_from_file_cache(index_name)
    if index is None:
        logging.info(f"=====> Build index from file!")
        documents = SimpleDirectoryReader(input_files=[file]).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.set_index_id(index_name)
        index.storage_context.persist()
        logging.info(
            f"=====> Save index to disk path: {index_cache_file_dir / index_name}")
    prompt = get_prompt_template(lang_code)
    logging.info('=====> Use llama file with chatGPT to answer!')
    logging.info('=====> dialog_messages')
    logging.info(dialog_messages)
    logging.info('=====> text_qa_template')
    logging.info(prompt)
    answer = index.as_query_engine(text_qa_template=prompt, similarity_top_k=5).query(dialog_messages)
    answer.response = remove_prompt_from_text(answer.response)
    total_embedding_token_count = token_counter.total_embedding_token_count
    total_llm_token_count = token_counter.total_llm_token_count

    token_counter.reset_counts()
    return answer, total_llm_token_count, total_embedding_token_count


def get_answer_from_llama_file_route_engine(messages, file):
    summary = 'summary_'
    keyword = 'keyword_'
    dialog_messages = format_dialog_messages(messages)
    lang_code = get_language_code(remove_prompt_from_text(messages[-1]))
    index_name = get_index_name_from_file(file)
    summary_index_name = summary+index_name
    keyword_index_name = keyword+index_name
    vector_index = get_index_from_file_cache(index_name)
    summary_index = get_index_from_file_cache(summary_index_name)
    keyword_index = get_index_from_file_cache(keyword_index_name)
    if vector_index is None or summary_index is None or keyword_index is None:
        documents = SimpleDirectoryReader(input_files=[file]).load_data()
        if vector_index is None:
            logging.info(f"=====> Build vector_index from file!")
            vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            vector_index.set_index_id(index_name)
            vector_index.storage_context.persist()
            logging.info(
                f"=====> Save index to disk path: {index_cache_file_dir / index_name}")
        if summary_index is None:
            logging.info(f"=====> Build summary_index from file!")
            summary_index = SummaryIndex.from_documents(documents,service_context=service_context)
            summary_index.set_index_id(summary_index_name)
            summary_index.storage_context.persist()

        if keyword_index is None:
            logging.info(f"=====> Build keyword_index from file!")
            keyword_index = SimpleKeywordTableIndex.from_documents(documents,service_context=service_context)
            keyword_index.set_index_id(keyword_index_name)
            keyword_index.storage_context.persist()

    prompt = get_prompt_template(lang_code)
    logging.info('=====> Use llama file with chatGPT to answer!')
    logging.info('=====> dialog_messages')
    # logging.info(dialog_messages)
    logging.info('=====> text_qa_template')
    logging.info(prompt)
    vector_query_engine = vector_index.as_query_engine(text_qa_template=prompt, similarity_top_k=5)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to the eassy or the file on "
            " What I Worked On."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context using vectors or Embedding from the essay or the file on What"
            " I Worked On."
        ),
    )

    keyword_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context using keywords from  the essay or the file on What"
            " I Worked On."
        ),
    )

    query_engine = RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
            keyword_tool
        ],
    )
    answer = query_engine.query(dialog_messages)
    selectors = ["summary index engine","vector index engine","keyword index engine"]
    inds = answer.metadata["selector_result"].inds
    engine_names = ''
    for index, ind in enumerate(inds):
        if index >= 1:
            engine_names += ' , '
        engine_names += selectors[ind]

    response = remove_prompt_from_text(answer.response)
    return_content = f"{response} \n --(search by this engine: {engine_names})"
    answer.response = return_content
    total_embedding_token_count = token_counter.total_embedding_token_count
    total_llm_token_count = token_counter.total_llm_token_count
    token_counter.reset_counts()
    return answer, total_llm_token_count, total_embedding_token_count


def get_text_from_whisper(voice_file_path):
    with open(voice_file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(file=f, model="whisper-1")
    return transcript.text


lang_code_voice_map = {
    'zh': ['zh-CN-XiaoxiaoNeural', 'zh-CN-XiaohanNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural'],
    'en': ['en-US-JennyNeural', 'en-US-RogerNeural', 'en-IN-NeerjaNeural', 'en-IN-PrabhatNeural', 'en-AU-AnnetteNeural',
           'en-AU-CarlyNeural', 'en-GB-AbbiNeural', 'en-GB-AlfieNeural'],
    'ja': ['ja-JP-AoiNeural', 'ja-JP-DaichiNeural'],
    'de': ['de-DE-AmalaNeural', 'de-DE-BerndNeural'],
}


def convert_to_ssml(text, voice_name=None):
    try:
        logging.info("=====> Convert text to ssml!")
        logging.info(text)
        text = remove_prompt_from_text(text)
        lang_code = get_language_code(text)
        if voice_name is None:
            voice_name = random.choice(lang_code_voice_map[lang_code])
    except Exception as e:
        logging.warning(f"Error: {e}. Using default voice.")
        voice_name = random.choice(lang_code_voice_map['zh'])
    ssml = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">'
    ssml += f'<voice name="{voice_name}">{text}</voice>'
    ssml += '</speak>'

    return ssml


def get_voice_file_from_text(text, voice_name=None):
    speech_config = SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.set_speech_synthesis_output_format(
        SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    speech_config.speech_synthesis_language = "zh-CN"
    file_name = f"{index_cache_voice_dir}{uuid.uuid4()}.mp3"
    file_config = AudioOutputConfig(filename=file_name)
    synthesizer = SpeechSynthesizer(
        speech_config=speech_config, audio_config=file_config)
    ssml = convert_to_ssml(text, voice_name)
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == ResultReason.SynthesizingAudioCompleted:
        logging.info("Speech synthesized for text [{}], and the audio was saved to [{}]".format(
            text, file_name))
    elif result.reason == ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        logging.info("Speech synthesis canceled: {}".format(
            cancellation_details.reason))
        if cancellation_details.reason == CancellationReason.Error:
            logging.error("Error details: {}".format(
                cancellation_details.error_details))
    return file_name


if __name__ == '__main__':
    print("----- start -----")
    # test for tts
    # text = "臣密言：臣以险衅，夙遭闵凶。生孩六月，慈父见背；行年四岁，舅夺母志。祖母刘愍臣孤弱，躬亲抚养。臣少多疾病，九岁不行，零丁孤苦，至于成立。既无叔伯，终鲜兄弟，门衰祚薄，晚有儿息。外无期功强近之亲，内无应门五尺之僮"
    # text = "我与父亲不相见已二年余了,我最不能忘记的是他的背影。那年冬天,祖母死了,父亲的差使也交卸了,正是祸不单行的日子。我从北京到徐州打算跟着父亲奔丧回家"
    # get_voice_file_from_text(text)
    # print("----- end tts ------")

    # test gpt
    # res, totalToken, total_embedding_model_tokens = get_answer_from_chatGPT("给我三个悲伤的成语")
    # print(res)
    # print(totalToken)

    # test speech whisper model
    # res = get_text_from_whisper("/Users/bobo/Downloads/audio_message.mp4")
    # print(res)

    # test for llama_index
    # documents = SimpleDirectoryReader(input_files=["/Users/bobo/Downloads/物料.txt"]).load_data()
    # index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # index.set_index_id("tset20230000000")
    # prompt = get_prompt_template()
    # answer = index.as_query_engine(text_qa_template=prompt, similarity_top_k=5).query("总结下这个内容")
    # answer.response = remove_prompt_from_text(answer.response)
    # for node in answer.source_nodes:
    #     if node.similarity != None:
    #         print(node.similarity)
    #         print(node.source_text)
    #
    # print(answer)


    # test for rrs
    # documents = []
    # urls = ["https://rsshub.app/zhihu/hotlist"]
    # rss_documents = RssReader().load_data(urls)
    # documents = documents + rss_documents
    # print(documents)

    # mss = []
    # dialog = "介绍下美国总统肯尼迪"
    # mss.append(f'User:{dialog}')
    # gpt_response = get_answer_from_chatGPT(mss)
    # print(gpt_response)
    # from server import insert_space
    # mss.append('chatGPT: %s' % insert_space(f'{gpt_response}'))
    # dialog = "我上一个对话是什么"
    # mss.append(f'User:{dialog}')
    # gpt_response = get_answer_from_chatGPT(mss)
    # print(gpt_response)
    # mss.append('chatGPT: %s' % insert_space(f'{gpt_response}'))
    # dialog = "他是第几届"
    # mss.append(f'User:{dialog}')
    # gpt_response = get_answer_from_chatGPT(mss)
    # print(gpt_response)



    # fileType = "txt"
    # temp_file_filename = index_cache_file_dir / ('物料'+"."+fileType)
    # temp_file_md5 = md5(temp_file_filename)
    # file_md5_name = index_cache_file_dir / (temp_file_md5 + '.' + fileType)
    # if not file_md5_name.exists():
    #     temp_file_filename.rename(file_md5_name)
    #
    # answer, total_llm_token_count, total_embedding_token_count = get_answer_from_llama_file_route_engine("詹姆斯做过哪些慈善和公益",file_md5_name)
    # print(answer)




