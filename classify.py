import asyncio
import pandas as pd
import openai
from worker_pool import WorkerPool
import os
from pydantic import BaseModel
from enum import Enum


# Local directory containing input parquet files (one per part)
# Expected filenames follow the pattern used by get_fname()
DATA_DIR = './wildchat-data'

# Local directory to write classified parquet outputs to
RESULTS_DIR = './wildchat-results'

# OpenAI model to use for classification
MODEL = 'gpt-4o-mini'



class IntentLabel(Enum):
    WEB_SITE_NAVIGATION = 'web_site_navigation'
    INFORMATION_LOOKUP = 'information_lookup'
    INFORMATION_GATHERING = 'information_gathering'
    TRANSLATION_OR_CONVERSION = "translation_or_conversion"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    IMAGE_CREATION = "image_creation"
    TEXT_GENERATION = "text_generation"
    OPEN_ENDED_DISCOVERY = "open_ended_discovery"


class IntentClassification(BaseModel):
    user_intent_explanation: str
    user_intent: IntentLabel


intent_prompt = '''<| start instructions |>
# Task Overview
You will be given a conversation between a user and an AI chatbot. Your job is to classify the user intent in the conversation. User intent is defined as the user's purpose for conversing with the AI agent. Note that the intent is to be based on the user's query and not necessarily on the AI agent's response.


# Task details
Your task is to fill out the following fields:
user_intent_explanation: Based on the provided conversation, explain in one sentence what the user intent is.
user_intent: Based on your explanation, provide one of the following labels:
- web_site_navigation: Conversations where the user simply wants to go to a particular website or service. For example, "facebook" indicates the user wants to go to the Facebook page to logon, "gmail" indicates the user wants to go to Gmail to perform email related actions and "google translate" indicates the user wants to go to "Google Translate" page.
- information_lookup: Conversations where the user wants to find factual information or answers to specific questions. The agent's responses are typically direct, concise, and informative, providing the relevant information and/or links to the sources.
- information_gathering: Conversations where the user wants to gather information that they can use to do their task. This could be asking the agent for recommendations, asking the agent to compare two or more objects, events, ideas, problems, or situations, etc.
- translation_or_conversion: Conversations where the user wants to convert information from one form or representation to another, such as from one language to another, from one unit to another, from words to numbers or vice versa.
- summarization: Conversations where the user wants to get a brief statement that captures the main idea or theme of the given information.
- analysis: Conversations where the user asks analytical or conceptual questions about a complex topic or problem. The user's questions require some degree of reasoning, interpretation, argumentation, comparison, and/or data processing from the agent. The agent's responses are typically explanatory and detailed, and are based on evidence, logic, and facts.
- image_creation: Conversations where the user asks the agent to either generate or modify an image based on specified criteria or constraints.
- text_generation: Conversations where the user asks the agent to either generate new content or modify existing content based on specified criteria or constraints. In the case of generating new content, the user's questions require some degree of creativity, novelty, or innovation from the agent. The agent's responses contain new or modified outputs that match the user's specifications.
- open_ended_discovery: Conversations where the user wants to chat or play with the agent out of curiosity, boredom, or humor, or else explore broad ideas or areas of interest without a specific goal or information need in mind. The agent's responses are typically suggestive and engaging. The agent may also encourage further inquiry or action from the user to deepen their discovery experience.


# Hints
Provide your answers in **English** using the given structured output format.
<| end instructions |>

<| start conversation |>
{content}
<| end conversation |>

<| end prompt |>
'''




class DomainLabel(Enum):
    COMPUTERS_AND_ELECTRONICS = 'computers_and_electronics'
    HOME_AND_AUTO = 'home_and_auto'
    PROGRAMMING_AND_SCRIPTING = 'programming_and_scripting'
    DATA_ANALYSIS_AND_VISUALIZATION = 'data_analysis_and_visualization'
    MACHINE_LEARNING_AND_AI = 'machine_learning_and_ai'
    ENGINEERING_AND_DESIGN = 'engineering_and_design'
    TRANSLATION_AND_LANGUAGE = 'translation_and_language'
    PHYSICS_AND_CHEMISTRY = 'physics_and_chemistry'
    BIOLOGY = 'biology'
    HEALTH_AND_MEDICINE = 'health_and_medicine'
    MATHEMATICS_AND_LOGIC = 'mathematics_and_logic'
    BUSINESS_AND_FINANCE = 'business_and_finance'
    MARKETING_AND_SALES = 'marketing_and_sales'
    EDUCATION_AND_LEARNING = 'education_and_learning'
    GAMING = 'gaming'
    ENTERTAINMENT = 'entertainment'
    SPORTS_AND_FITNESS = 'sports_and_fitness'
    HISTORY_AND_CULTURE = 'history_and_culture'
    LAW_AND_POLITICS = 'law_and_politics'
    RELIGION_AND_PHILOSOPHY = 'religion_and_philosophy'
    FASHION_AND_BEAUTY = 'fashion_and_beauty'
    FOOD_AND_DRINK = 'food_and_drink'
    TRAVEL_AND_TOURISM = 'travel_and_tourism'
    SMALL_TALK_AND_CHATBOT = 'small_talk_and_chatbot'
    SHOPPING_AND_ECOMMERCE = 'shopping_and_ecommerce'
    JOBS_AND_EMPLOYMENT = 'jobs_and_employment'
    CREATIVE_WRITING_AND_EDITING = 'creative_writing_and_editing'
    PROFESSIONAL_WRITING_AND_EDITING = 'professional_writing_and_editing'
    ADULT = 'adult'
    OTHER = 'other'

class DomainClassification(BaseModel):
    conversation_domain_explanation: str
    conversation_domain: DomainLabel


domain_prompt = '''<| start instructions |>
# Task Overview
You will be given a conversation between a user and an AI chatbot. Your job is to classify the topical domain of the conversation. The topical domain of a conversation is the general subject area the conversation falls under.


# Task details
Your task is to fill out the following fields:
conversation_domain_explanation: Based on the provided conversation, explain in one sentence what the domain of the conversation is.
conversation_domain: Based on your explanation, provide one of the following labels:
- computers_and_electronics: The user's topical domain is related to general questions, comments or discussion of computer and electronics, including technology products and services, and the internet.
- home_and_auto: The user's topical domain is related to home and auto related projects, products, services, and repairs.
- programming_and_scripting: The user's topical domain is related to creating, modifying or debugging computer code, programs, websites, web applications or scripts using various languages, frameworks or tools.
- data_analysis_and_visualization: The user's topical domain is related to collecting, analyzing, visualizing or presenting data using various methods, tools or software.
- machine_learning_and_ai: The user's topical domain is related to applying or developing artificial intelligence or machine learning techniques or models using various methods, tools or software.
- engineering_and_design: The user's topical domain is related to the application of engineering principles or methods to various fields, such as civil, mechanical, electrical, chemical, etc., or to the design or creation of visual or auditory works.
- translation_and_language: The user's topical domain is related to the translation of words, phrases or sentences in different languages, or to the learning, study or practice of word meanings or the structure, function, evolution, or diversity of languages or linguistic features, rules or systems.
- physics_and_chemistry: The user's topical domain is related to the study of the matter, energy, forces, reactions or interactions of the physical or chemical world, or to the concepts, theories or experiments of various branches of physics or chemistry.
- biology: The user's topical domain is related to the study of the structure, function, evolution or diversity of living organisms.
- health_and_medicine: The user's topical domain is related to the diagnosis, treatment or prevention of diseases or disorders, or to the physical or mental well-being of humans or animals.
- mathematics_and_logic: The user's topical domain is related to the application or solution of mathematical or logical problems or puzzles, such as arithmetic, algebra, geometry, calculus, etc., or to the concepts, theories or arguments of various branches of mathematics or logic.
- business_and_finance: The user's topical domain is related to the operation, strategy or analysis of an organization or an industry, or to the study of the production, distribution or consumption of goods or services, or to the market, trade or currency issues, or to financial transaction, products, services or regulations.
- marketing_and_sales: The user's topical domain is related to promoting or selling products or services, or to creating, evaluating or improving advertisements or marketing strategies or campaigns, or to persuading or influencing potential customers or clients.
- education_and_learning: The user's topical domain is related to learning, teaching or studying various subjects or skills, or to the methods, tools or resources for education or learning, or to the educational institutions, programs or policies.
- gaming: The user's topical domain is related to the playing, designing or reviewing of video games.
- entertainment: The user's topical domain is related to or is about entertainment media such as movies, tv shows, music, or books, or the celebrities in that media.
- sports_and_fitness: The user's topical domain is related to the physical activities, exercises or games that involve skill, strength or endurance, or to the equipment, rules or strategies of sports or fitness, or to the sports teams, players or events.
- history_and_culture: The user's topical domain is related to the past or present events, people, places, customs, beliefs, values or arts of a group of people or a region, or to the appreciation, criticism or creation of historical or cultural works.
- law_and_politics: The user's topical domain is related to the rules, regulations or principles that govern the conduct or relations of people or entities, or to the rights, responsibilities or remedies of legal subjects or cases, or to the opinions, issues or actions of political actors or institutions.
- religion_and_philosophy: The user's topical domain is related to the belief, worship or practice of a supernatural or divine power or entity, or to the personal or philosophical quest for meaning or purpose, or to the concepts, theories or arguments of various schools of thought.
- fashion_and_beauty: The user's topical domain is related to the style, appearance or attractiveness of clothing, accessories, cosmetics or hair, or to the trends, tips or advice of fashion or beauty, or to the fashion or beauty products, services or brands.
- food_and_drink: The user's topical domain is related to the preparation, storage or consumption of food or beverages, or to the recipes, ingredients, flavors or health benefits of food or beverages, or to the preferences, opinions or recommendations of food or beverages.
- travel_and_tourism: The user's topical domain is related to the movement, exploration or discovery of different places or cultures, or to the transportation, accommodation or activities of travelers, or to the travel guides, tips or advice.
- small_talk_and_chatbot: The user's topical domain is related to general questions or answers about the AI agent or the user is engaging the AI agent in jokes or other casual interaction that is not about any of the topical domains. If the user is talking about any of the topical domains, even if it appears they are confused or joking, then the topical domain should be that domain and not 'Small talk and chatbot'.
- shopping_and_ecommerce: The user's topical domain is related to purchasing goods or services, including products, prices, discounts, reviews, or points.
- jobs_and_employment: The user's topical domain is related to researching, understanding, seeking, or interviewing for a job or employment, including resume writing.
- creative_writing_and_editing: The user's topical domain is related to the creation, improvement or evaluation texts for creative or imaginative purposes or audiences, such as stories, poems, songs, etc., or to the genres, styles or themes of creative writing.
- professional_writing_and_editing: The user's topical domain is related to the creation, improvement or evaluation of texts for academic purposes or audiences, such as essays, articles, theses, etc., or to the citation, referencing or plagiarism issues of academic writing.
- adult: The user's topical domain is related to searching for, consuming, or producing pornographic content.
- other: the user's topical domain does not fit into any of the above specified topical domains. The topical domain should be 'Other' if and only if it absolutely does not fit into any of the other topical domains.


# Hints
Provide your answers in **English** using the given structured output format.
<| end instructions |>

<| start conversation |>
{content}
<| end conversation |>

<| end prompt |>
'''



class CompletionStatus(Enum):
    NOT_COMPLETED = 'not_completed'
    PARTIALLY_COMPLETED = 'partially_completed'
    COMPLETED = 'completed'

class InteractionAnswer(BaseModel):
    user_task_summary: str
    completed_explanation: str
    completed: CompletionStatus
    explicit_pos_feedback_explanation: str
    explicit_pos_feedback: bool
    explicit_neg_feedback_explanation: str
    explicit_neg_feedback: bool



task_prompt = '''<| start instructions |>
# Task Overview
You will be given a conversation between a user and an AI chatbot. You will summarize the main task that the user is trying to accomplish in the conversation. You will also determine whether the AI chatbot is able to complete the task and whether any explicit positive or negative feedback is given by the user during the conversation.


# Task details
Your task is to fill out the following fields:
user_task_summary: A one sentence summary of the main task the user is trying to accomplish in this conversation. Include only the task the **user** intends to complete, and not the one the agent performs.
completed_explanation: Based on the provided conversation, explain in one sentence whether the AI chatbot completed the user's task.
completed: Based on your explanation, provide one of the following labels:
- not_completed: The AI chatbot did not make substantive progress towards completing the user's task.
- partially_completed: The AI chatbot made progress towards completing the user's task, but did not complete it.
- completed: The AI chatbot completed the user's task.
explicit_pos_feedback_explanation: Based on the provided conversation, explain in one sentence whether the user provided explicit positive feedback for the AI chatbot. Explicit positive feedback may include thanking or complimenting the chatbot.
explicit_pos_feedback: Based on the explanation, say True if the user did provide explicit positive feedback for the AI chatbot and False otherwise.
explicit_neg_feedback_explanation. Based on the provided conversation, explain in one sentence whether the user provided explicit negative feedback for the AI chatbot. Explicit negative feedback may include correcting the chatbot's mistakes or expressing dissatisfaction, frustration, annoyance, or anger with the AI chatbot.
explicit_neg_feedback: Based on the explanation, say True if the user did provide explicit negative feedback for the AI chatbot and False otherwise. 


# Hints
Provide your answers in **English** using the given structured output format.
<| end instructions |>

<| start conversation |>
{content}
<| end conversation |>

<| end prompt |>
'''



def abbreviate_turn(content, max_len):
    if len(content) < max_len:
        return content

    return content[:max_len//2] + ' [... more text ...] ' + content[-max_len//2:]

def create_conversations(part_df):
    author_to_label = {'user': 'user', 'assistant': 'agent'}

    def make_convo_str(convo):
        return '\n\n'.join(f"<| start {author_to_label[turn['role']]} message |>\n{abbreviate_turn(turn['content'], 5000)}\n<| end {author_to_label[turn['role']]} message |>" for turn in convo)
    
    part_df['Text'] = part_df['conversation'].apply(make_convo_str)

    return part_df


def get_fname(part, dataset):
    if dataset == 'wildchat-1m':
        return f'{DATA_DIR}/{dataset}/train-{part}-of-00014.parquet'
    elif dataset == 'wildchat-4.8m':
        return f'{DATA_DIR}/{dataset}/train-{part}-of-00086.parquet'

    return None 


def get_data_by_part(part, dataset):
    print(f"Retrieving data for part {part} of {dataset}...")
    part_df = pd.read_parquet(get_fname(part, dataset))

    part_exchange_df = create_conversations(part_df)

    return part_exchange_df


def get_async_openai_client():
    # Reads the API key from the OPENAI_API_KEY environment variable.
    return openai.AsyncOpenAI(max_retries=3)

def classify(part_exchange_df, prompt, answer_format):
    print("Running workers...")
    client = get_async_openai_client()
    worker_pool = WorkerPool(client, MODEL, 20)

    convos = part_exchange_df["Text"].tolist()

    task_prompts = {i:prompt.format(content=convo) for i, convo in enumerate(convos)}
    results = asyncio.run(worker_pool.get_chat_completions(task_prompts, answer_format))

    failed_keys = [
        key for key, _, completion, error in results
        if error is not None
        or completion is None
        or not completion.get('choices')
        or completion['choices'][0]['message'].get('parsed') is None
    ]
    if failed_keys:
        first_failed_keys = ", ".join(str(key) for key in failed_keys[:10])
        raise RuntimeError(
            f"{len(failed_keys)} classification requests failed (first keys: {first_failed_keys}). "
            "The result file was not written; rerun this shard to retry."
        )
    
    completion_dict = {key: completion['choices'][0]['message']['parsed'] if completion is not None and len(completion['choices']) > 0 else None for key, prompt, completion, error in results}
    completions = [completion_dict[i] for i in range(len(convos))]

    for key in answer_format.model_fields:
        part_exchange_df[key] = pd.Series([c[key] if c is not None else None for c in completions])
    
    return part_exchange_df



if __name__ == '__main__':
    # # Wildchat 1m parts 00000-00013
    # dataset = 'wildchat-1m'
    # parts = [f'{i}'.zfill(5) for i in range(14)]

    # Wildchat 4.8m parts 00000-00085
    dataset = 'wildchat-4.8m'
    parts = [f'{i}'.zfill(5) for i in range(86)]

    results_dir = f'{RESULTS_DIR}/{dataset}-results'
    os.makedirs(results_dir, exist_ok=True)

    for part in parts:
        print("Processing part " + str(part) + "...")

        # check if results already exist
        domain_path = f'{results_dir}/{dataset}-{part}-domain.parquet'
        intent_path = f'{results_dir}/{dataset}-{part}-intent.parquet'
        task_path = f'{results_dir}/{dataset}-{part}-task.parquet'

        domain_exists = os.path.exists(domain_path)
        intent_exists = os.path.exists(intent_path)
        task_exists = os.path.exists(task_path)

        if domain_exists and intent_exists and task_exists:
            print(f"Results for part {part} already exist, skipping...")
            continue

        part_exchange_df = get_data_by_part(part, dataset)

        # classify domain
        if not domain_exists:
            part_conv_data_df = classify(part_exchange_df, domain_prompt, DomainClassification)
            part_conv_data_df['conversation_domain'] = part_conv_data_df['conversation_domain'].astype(str)
            part_conv_data_df.to_parquet(domain_path)
        else:
            print(f"Domain results for part {part} already exist, skipping...")
        
        # classify intent
        if not intent_exists:
            part_conv_data_df = classify(part_exchange_df, intent_prompt, IntentClassification)
            part_conv_data_df['user_intent'] = part_conv_data_df['user_intent'].astype(str)
            part_conv_data_df.to_parquet(intent_path)
        else:
            print(f"Intent results for part {part} already exist, skipping...")

        # classify task
        if not task_exists:
            part_conv_data_df = classify(part_exchange_df, task_prompt, InteractionAnswer)
            part_conv_data_df['completed'] = part_conv_data_df['completed'].astype(str)
            part_conv_data_df.to_parquet(task_path)
        else:
            print(f"Task results for part {part} already exist, skipping...")