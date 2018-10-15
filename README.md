# Memory-based Story Learning  for Conversational Question Answering.

it is based on End-to-End Memory Networks. And I extend this model using Skip-thought Vector for dealing with sentence answer. 

## Dataset

We use the "PandaQA" dataset. Unfortunately, it is not available due to the Licence issues of the owner. the PandaQA dataset is very similar with the ![Alt text](https://research.fb.com/downloads/babi/ "bAbI dataset") of QA tasks.

## Model

The objective function of this model is to learn the representation of the answer sentence according to the question. The skip-thoughts vector is used as a sentence encoder. And we can infer the answer representation with the End-to-End Memory Networks architecture. The final answer is choosed as the nearest answer candidate on training dataset.

You can check the details about the model on ![Alt text](/article/KCC2016_article_final.pdf, "the article file"), which is a Korean version article and the English version is on preparing.

![Alt text](/article/model.jpg?raw=true "Extended End-to-End Memory Network")

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
