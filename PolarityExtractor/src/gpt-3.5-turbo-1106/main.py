import logging
import pandas as pd
import openai
import time
import os
import json
import ast

import prompts

from langchain import OpenAI
from langchain.prompts import PromptTemplate

os.environ['OPENAI_API_KEY'] = 'your API key'

logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',)

logger = logging.getLogger(__name__)


def main():
    N_USERS = 500
    N_REVIEWS = 20
    N_RETRY = 3
    RETRY_WAITING_TIME = 60
    prompt_type = "zero-shot-ja"
    model_name = "gpt-3.5-turbo-1106"
    temperature = 0.0
    result_path = f"./results/{prompt_type}/user-{
        N_USERS}_reviews-{N_REVIEWS}_turbo-1106.json"
    columns = ['item_id', 'user_id', 'story', 'casting', 'direction',
               'movie', 'music', 'overall', 'created', 'title', 'review']

    users_review_data = pd.read_csv(
        f"../../movie/data/review_data_polarity_u-{N_USERS}_i-{N_REVIEWS}.csv", names=columns)
    logger.info(f"Downloaded users_review_data: {users_review_data.shape}")

    logger.info("Begin Extracting Polarity Data")
    start_time = time.time()
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for index in range(0, N_USERS * N_REVIEWS):
            logger.info(f"*********** index: {index} ***********\n")
            logger.info(
                f"### user_id: {users_review_data.iloc[index]['user_id']} ###")
            logger.info(
                f"### item_id: {users_review_data.iloc[index]['item_id']} ###")

            item_id = users_review_data.iloc[index]['item_id']
            user_id = users_review_data.iloc[index]['user_id']
            review = users_review_data.iloc[index]['review']

            for i in range(N_RETRY):
                try:
                    llm = OpenAI(model_name=model_name,
                                 temperature=temperature)
                    logger.info(
                        f"Set OpenAI model: model={model_name}, temperature={temperature}")
                    chain_result = prompts.PolarityExtract(promtp_type=prompt_type,
                                                           llm=llm,
                                                           review=review,
                                                           verbose=False)
                    logger.info(f"finished extract {index}.")
                    break
                except Exception as e:
                    logger.info(
                        f"post_chatgpt failed with exception: {e}. retry: {i+1}")
                    time.sleep(RETRY_WAITING_TIME)
                    continue
            else:
                logger.info("post_chatgpt failed. skip this review")
                chain_result = """
                    {
                        {"story": "None"},
                        {"casting": "None"},
                        {"direction": "None"},
                        {"images": "None"},
                        {"music": "None"},
                    }
                """
                continue

            chain_result = chain_result.replace("{\n", "[").replace("\n}", "]")
            logger.info(f"chain_result: {chain_result}")
            data_list = ast.literal_eval(chain_result)
            data_for_json = {
                "user_id": user_id,
                "item_id": str(item_id),
                "story": data_list[0]["story"],
                "casting": data_list[1]["casting"],
                "direction": data_list[2]["direction"],
                "images": data_list[3]["images"],
                "music": data_list[4]["music"],
            }
            logger.info(f"data_for_json: {data_for_json}")
            json.dump(data_for_json, f, indent=4)

            if index == N_USERS * N_REVIEWS - 1:
                f.write("\n")
            else:
                f.write(",\n")

        f.write("]\n")
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Finished Extracting Polarity Data: {result_path}")
    logger.info(f"execution time: {execution_time:.2f} (sec)")


if __name__ == '__main__':
    main()
