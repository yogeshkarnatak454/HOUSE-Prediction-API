import os
from dotenv import load_dotenv

from groq import Groq
#  load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_prediction(area,bedrooms,bathrooms,age,predicted_price):

    prompt = f"""
      You are a real estate assistant
      House Details:

      -Area:{area} sq ft
      -Bedrooms:{bedrooms}
      -Bathrooms:{bathrooms}
      - Age:{age} years


      Predicted Price: ${predicted_price}


     Tasks: 
     1 Explain in simple language why this house has this predicted_price
     2 Give 3 suggestions  to increase the value of the house 
     3 keep the explaination under 159 words




"""
    
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages =[
            {

                "role":"user",
                "content":prompt
            }


        ],
        temperature= 0.5,
        max_tokens = 300

    )

    return response.choices[0].message.content