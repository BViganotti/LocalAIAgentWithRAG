This is an existing project that i want to modify. 
I want to modify it to build an inspiration base of high impact social media posts serving as inspiration for further creation.
Currently i tries to read a csv file that doesn't exist any more, my data is in the data/ directory and is json or txt data.
Also i do not want it to be a chatbot, i want it to be an agent that can generate article for Reddit and LinkedIn and Twitter post given a specific topic fetched from the topic directory as a file (could be pdf, txt, md files). 
It should write the articles/posts about the topic while using the inspiration base to make high impact content.

I want to replace chromadb with qdrant
i want qdrant to work in a container
I want to store the articles in a postgresql database that will also be in a container
No standard ports for postgres please, it's probably already taken.
