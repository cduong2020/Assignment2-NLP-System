from retriever import Retriever
from transformers import pipeline

# Initialize components
# to get the relevant information for the query
retriever = Retriever()
# loads TinyLlama
# wrapper to the complex mathematics code that created the output text from a prompt
# the prompt is created from a query and the data points closet to the query in the space
generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def run_rag():
    # get the questions, open the file to write the answers to
    with open("questions.txt", "r", encoding="utf-8") as q_file, \
            open("system_output.txt", "w", encoding="utf-8") as out_file:

        for line in q_file:
            # get rid of extra spaces
            question = line.strip()
            # skip empty lines
            if not question: continue

            # 1. RETRIEVE: k=1 is safest for small models to avoid name-mixing
            # This is the R in RAG
            # grab the closest bit of data to the question
            context_chunks = retriever.search(question, k=1)
            # remove extra spaces
            context_text = context_chunks[0].strip()

            # 2. Augment
            # This is the a in RAG
            # gives what the LM need to craft an answer
            prompt = (
                f"<|user|>\n"
                f"Use ONLY the following context to answer the question. "
                f"Do not use any outside information. If the answer isn't there, say 'Information not found'.\n\n"
                f"Context: {context_text}\n"
                f"Question: {question}\n<|assistant|>\n"
            )

            # 3. GENERATION: temperature=0.0 is the "Gold Standard" for facts
            # This is the G in RAG
            res = generator(prompt,
                            max_new_tokens=150,# max characters for the output text
                            temperature=0.01,# risk level, how much randomness/creativity to use -- low
                            do_sample=True,# samples from top choice words
                            repetition_penalty=1.2)  # prevents the model from using the same word/phrase twice in a given time frame
            # clean up the answer
            answer = res[0]['generated_text'].split("<|assistant|>")[-1].strip()
            # Only keep the first relevant sentence
            answer = answer.split("\n")[0].split("Question:")[0].strip()

            # write the answer to file with the question
            out_file.write(f"Q: {question}\nA: {answer}\n\n")


if __name__ == "__main__":
    run_rag()