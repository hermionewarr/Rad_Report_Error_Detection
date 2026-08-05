#evaluate
#litle test
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity

fill_mask = pipeline(
    "fill-mask",
    model="./bert-pretrained2",
    tokenizer=tokenizer
)

print(fill_mask("Extremely FDG-avid mid-oesophageal [MASK].")) #tumour
print(fill_mask("Extremely FDG-avid mid-[MASK] tumour.")) #tumour

#evaluate on test data
#eval_results = trainer.evaluate()
#print(eval_results)

#plot visualisation
def normalize(x):
    return x / x.norm(dim=1, keepdim=True)

def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    cleaned_texts = []
    for text in texts:
        if pd.isna(text):
            cleaned_texts.append("")
        else:
            cleaned_texts.append(str(text).strip())

    inputs = tokenizer(
        cleaned_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.bert(**inputs)

    # CLS token = first token
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu()

def build_corpus_embeddings(texts, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = get_embeddings(batch)
        all_embeddings.append(emb)

    return torch.cat(all_embeddings, dim=0)



emb = get_embeddings(test_texts).numpy()

kmeans = KMeans(n_clusters=5).fit(emb)


emb_2d = TSNE(n_components=2).fit_transform(emb)

plt.scatter(emb_2d[:,0], emb_2d[:,1])
plt.savefig("bert_plot2.png") #looks shit, looks better with 50 pochs


def retrieve(query, corpus_embeddings, top_k=5):
    q_emb = get_embeddings([query])  # shape: (1, hidden)
    corpus_embeddings = normalize(corpus_embeddings)
    q_emb = normalize(q_emb)
    sims = cosine_similarity(
        q_emb, 
        corpus_embeddings, 
        dim=1
    )

    topk = torch.topk(sims, k=top_k)

    return topk.indices, topk.values


corpus_embeddings = build_corpus_embeddings(test_texts)

query = test_texts[0]

indices, scores = retrieve(query, corpus_embeddings)

print("query:\n", query)

for i, s in zip(indices, scores):
    print("Score:", s.item())
    print(texts[i])
    print("-" * 50)

findings_texts = test_data["Findings"].fillna("").astype(str).str.strip().tolist()
conclusion_texts = test_data["Conclusion"].fillna("").astype(str).str.strip().tolist()

findings_emb = get_embeddings(findings_texts)
conclusion_emb = get_embeddings(conclusion_texts)
findings_emb = normalize(findings_emb)
conclusion_emb = normalize(conclusion_emb)
sims = cosine_similarity(findings_emb, conclusion_emb, dim=1)

print("Mean similarity:", sims.mean().item())

def alignment_accuracy(findings_emb, conclusion_emb, top_k=1):
    correct = 0

    for i in range(len(findings_emb)):
        sims = cosine_similarity(
            findings_emb[i].unsqueeze(0),
            conclusion_emb,
            dim=1
        )

        topk = torch.topk(sims, k=top_k).indices

        if i in topk:
            correct += 1

    return correct / len(findings_emb)

acc = alignment_accuracy(findings_emb, conclusion_emb, top_k=1)
print("Top-1 accuracy:", acc)

acc = alignment_accuracy(findings_emb, conclusion_emb, top_k=5)
print("Top-5 accuracy:", acc)

acc = alignment_accuracy(findings_emb, conclusion_emb, top_k=10)
print("Top-10 accuracy:", acc)

acc = alignment_accuracy(findings_emb, conclusion_emb, top_k=100)
print("Top-100 accuracy:", acc)