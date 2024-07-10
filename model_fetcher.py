from typing import Dict
import numpy as np
import pandas as pd
import string
import json
import requests
from bs4 import BeautifulSoup

class HuggingFaceModelFetcher():
    """
    A class to fetch model details from Hugging Face URLs and return as a Pandas DataFrame.

    Attributes:
    - url_to_parse (str): URL to parse for model details.
    - close_time (int), default = 10: Timeout duration in seconds for the HTTP request.

    Methods:
    - fetch_model_details(): Fetches model details from the specified URL.
    - show_help(): Prints helpful notes and links related to various embeddings and models.
    """

    def __init__(self,url_to_parse:str,close_time:int = 10):
        """
        Initialize the HuggingFaceModelFetcher with the URL and timeout duration.

        Args:
        - url_to_parse (str): URL to parse for model details.
        - close_time (int, optional), default = 10: Timeout duration in seconds for the HTTP request. Default is 10 seconds.
        """
        self.url_to_parse = url_to_parse
        self.close_time = close_time
        self.check_packages()

    def __repr__(self):
        return f"HuggingFaceModelFetcher()"

    def __str__(self):
        return "Class to fetch huggingface models and sort based on downloads and likes."

  def check_packages(self)->None:
    """
    Checks for required Python packages and installs them if not already installed.

    Returns:
    - None
    """
    !pip install --quiet importlib
    import importlib

    req_packages:list = ['typing','numpy','pandas','string','json','requests','bs4']

    for package_name in req_packages:
      try:
        importlib.import_module(package_name)
      except:
        try:
          !pip install --quiet {package_name}
        except Exception as e:
          print(f"Required package {package_name} was not installed!: {str(e)}")
    del importlib
    print("All required packages are installed.")
    return None

    def show_help(self)->None:
        """
        Prints helpful notes and links related to various embeddings and models.
        """
        help_content = """
        Notes:

        Gensim API Models
        # https://radimrehurek.com/gensim/models/word2vec.html

        ERNIE Embedding
        https://huggingface.co/docs/transformers/en/model_doc/ernie

        FLAIR Embedding
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
        https://flairnlp.github.io/docs/tutorial-embeddings/flair-embeddings

        LASER Embedding
        https://pypi.org/project/laserembeddings/

        HuggingFace Models
        https://huggingface.co/openai-community
        https://huggingface.co/google-bert
        https://huggingface.co/FacebookAI
        https://huggingface.co/distilbert
        https://huggingface.co/albert
        https://huggingface.co/google
        https://huggingface.co/microsoft
        https://huggingface.co/allenai
        https://huggingface.co/xlnet
        https://huggingface.co/flair
        https://huggingface.co/nghuyong
        https://huggingface.co/sentence-transformers
        """

        print(help_content.strip())
        return None

    def get_model_details(self,parsed_response:Dict)->pd.DataFrame:
        """
        Extracts model details from parsed JSON response and returns a Pandas DataFrame.

        Args:
        - parsed_response (dict): Parsed JSON response containing model information.

        Returns:
        - pd.DataFrame: DataFrame with columns Model_Name,Likes,Downloads,and Pipeline_Tag.
        """
        model_name_list,likes_list,downloads_list,pipeline_tag_list = [],[],[],[]

        for item in parsed_response['repos']:
            model_name_list.append(item.get('id','NAME NOT FOUND'))
            likes_list.append(int(item.get('likes',0)))
            downloads_list.append(int(item.get('downloads',0)))
            pipeline_tag_list.append(item.get('pipeline_tag',''))

        model_df = pd.DataFrame({
            'Model_Name':model_name_list,
            'Likes':likes_list,
            'Downloads':downloads_list,
            'Pipeline_Tag':pipeline_tag_list,
        })

        model_df = model_df[model_df['Downloads'] > 0]
        model_df = model_df[model_df['Model_Name'] != 'NAME NOT FOUND']
        model_df = model_df.sort_values(by=['Downloads','Likes'],ascending=[False,False])

        return model_df

    def get_model_information(self,parsed_response:BeautifulSoup)->pd.DataFrame:
        """
        Parses BeautifulSoup object to extract model details.

        Args:
        - parsed_response (BeautifulSoup): Parsed HTML content.

        Returns:
        - pd.DataFrame: DataFrame with aggregated model details.
        """
        all_parsed_responses = []

        try:
            for div_element in parsed_response.find_all('div',class_='SVELTE_HYDRATER contents'):
                data_props = div_element.get('data-props')

                if data_props:
                    props_dict = json.loads(data_props)

                    if props_dict.get('repos'):
                        all_parsed_responses.append(props_dict)

        except Exception as e:
            print(f'Error in get_model_information:{e}')
            pass

        models_df = pd.DataFrame()

        for item in all_parsed_responses:
            try:
                models_df = pd.concat([models_df,self.get_model_details(parsed_response=item)])

            except Exception as e:
                continue

        return models_df

    def fetch_model_details(self)->pd.DataFrame:
        """
        Fetches model details from the specified URL and returns as a Pandas DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with aggregated model details (Model_Name, Likes, Downloads, Pipeline_Tag).
        """

        try:
            url_response = requests.get(url=self.url_to_parse,timeout=self.close_time)

            if url_response.status_code == 200:
                parsed_response = BeautifulSoup(url_response.content,'html.parser')
                return self.get_model_information(parsed_response)

            else:
                print(f'Failed to retrieve content. Status code:{url_response.status_code}')
                return pd.DataFrame()

        except requests.exceptions.Timeout:
            print(f'Timeout error:The request timed out after {self.close_time} seconds.')
            return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            print(f'Request error:{e}')
            return pd.DataFrame()

# Example usage:
if __name__ == "__main__":
    hf_model_fetcher = HuggingFaceModelFetcher(url_to_parse='https://huggingface.co/allenai',close_time=10)
    print(hf_model_fetcher.show_help())
    model_results_df = hf_model_fetcher.fetch_model_details()
    print(model_results_df)

