import dspy
lm = dspy.LM("openai/gpt-4o-mini", api_key="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", api_base="https://openrouter.ai/api/v1")
dspy.configure(lm=lm)

def evaluate_math(expression: str):
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer)