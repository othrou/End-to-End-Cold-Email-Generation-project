import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        # Crée un modèle de prompt à partir d'un template
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        # Chaîne le prompt avec le modèle de langage (LLM)
        chain_extract = prompt_extract | self.llm
        
        # Exécute la chaîne avec le texte nettoyé en entrée
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        
        # Récupère le contenu de la réponse
        resultat = res.content
        
        try:
            # Initialise le parser JSON
            json_parser = JsonOutputParser()
            
            # Parse le résultat en JSON
            json_res = json_parser.parse(resultat)
            
            # Sélectionne le premier job trouvé (commenter cette ligne pour générer des mails pour tous les jobs disponibles)
            wanted_job = json_res[0]  # just comment this line dear friend in order to generate mails for all the jobs disponible in the page 

        except OutputParserException:
            # Lève une exception si le parsing échoue
            raise OutputParserException("Context too big. Unable to parse jobs.")
        
        # Retourne le job trouvé (ou une liste de jobs si multiple)
        return wanted_job if isinstance(wanted_job, list) else [wanted_job]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Othmane Y, a business development executive at OthRou International company. OthRou is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of OthRou 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            Remember you are Othmane Y, BDE at OthRou, here's my phone nbr +212 05 xx xx xx . 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":

    print(os.getenv("GROQ_API_KEY"))