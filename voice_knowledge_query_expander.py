from typing import List
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate


class QueryExpander:
    """
    A class to expand a single query into multiple semantically similar variations
    to improve retrieval & context information.
    """

    def __init__(self, temperature: float = 0.1):
       
        self.llm = ChatOpenAI(api_key=<ADD YOUR OPENAI_API_KEY>, temperature=temperature, model="gpt-5.4")

        # Prompt template for query expansion
        
        template= """You are a highly knowledgeable assistant. 
                    Given the following question, generate 3 different versions of the question 
                    that capture different aspects and perspectives of the original question. 
                    Make the variations semantically diverse but relevant.
                    
                    Original Question: {question}
                    
                    Generate variations in the following format:
                    1. [First variation]
                    2. [Second variation]
                    3. [Third variation]
                    
                    Only output the numbered variations, nothing else."""
                    
        self.query_expansion_prompt = PromptTemplate(
            input_variables=list("question"),
            template=template
        )

    def expand_query(self, question: str) -> List[str]:
        """
        Expand a single query into multiple variations.

        Args:
            question: The original question to expand.

        Returns:
            List of query variations including the original question.
        """
        try:
            # Get variations from LLM
            response = self.llm.invoke(
                self.query_expansion_prompt.format(question=question)
            )

            # Parse numbered list from response
            variations = [
                line.split(". ")[1] for line in response.content.strip().split("\n")
            ]

            variations.append(question)

            return variations

        except Exception as e:
            print(f"Error in query expansion: {e}")
            return [question]
