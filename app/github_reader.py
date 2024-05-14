import logging
import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.github import GithubRepositoryReader, GithubClient

from app.gpt import remove_prompt_from_text, format_dialog_messages, token_counter

github_token = os.environ.get("GITHUB_TOKEN")
owner = "comfyanonymous"
repo = "ComfyUI"
branch = "master"

model_name = "gpt-3.5-turbo"
llm = OpenAI(model_name=model_name)
github_client = GithubClient(github_token=github_token)
github_storage_context = StorageContext.from_defaults()


def get_answer_from_github(query_message):
    index_name = get_index_name_for_github(owner, repo)
    index = get_index_from_file_cache(index_name)
    dialog_messages = format_dialog_messages(query_message)
    if index is None:
        documents = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            # filter_directories=(
            #     ["docs"],
            #     GithubRepositoryReader.FilterType.INCLUDE,
            # ),
            filter_file_extensions=(
                [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".svg",
                    ".ico",
                    ".ipynb"
                ],
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
        ).load_data(branch=branch)
        index = VectorStoreIndex.from_documents(documents, storage_context=github_storage_context)
        index.set_index_id(index_name)
        index.storage_context.persist()

    query_engine = index.as_query_engine(llm=llm)
    answer = query_engine.query(
        dialog_messages
        # verbose=True,
    )
    answer.response = remove_prompt_from_text(answer.response)
    total_embedding_token_count = token_counter.total_embedding_token_count
    total_llm_token_count = token_counter.total_llm_token_count
    token_counter.reset_counts()
    return answer, total_llm_token_count, total_embedding_token_count


def get_index_name_for_github(owner, repo):
    return f"{owner}_{repo}"


def get_index_from_file_cache(name):
    try:
        index = load_index_from_storage(github_storage_context, index_id=name)
    except Exception as e:
        logging.error(e)
        return None
    return index
