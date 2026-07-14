from llm_service import explain_prediction

result = explain_prediction(

    area = 2000,
    bedrooms= 3,
    bathrooms=3,
    age=5,
    predicted_price= 82500000

)

print(result)