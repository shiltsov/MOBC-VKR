import openai
import logging

logger = logging.getLogger(__name__)

def request_llm(prompt, API_KEY, temperature=0.5, max_tokens=1000, model='gpt-4o-mini', platform='openai', response_type="text"):
    try:
        if platform == "deepseek":
            client = openai.OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
        else:
            client = openai.OpenAI(api_key=API_KEY)    

        response = client.chat.completions.create(
            model=model,             
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={'type': response_type}, 
            temperature=temperature,
            max_tokens=max_tokens            
        )      
        rsp = response.choices[0].message.content.strip()
        logger.info(f"LLM RESPONSE: {rsp}")
        return rsp        

    except Exception as e:
        logger.error(f"LLM ERROR: {e}")
        return None
