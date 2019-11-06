import cluster_keywords as ck
import numpy as np
import json
import csv
import re
import pdb

print("import ok!")

model = ck.load_FT_model("/home/ew-shopp/fasttext/models/cc.de.300.bin")

print("load_FT_embeddings ok!")

keywords = ["one", "one two", "one two three"]
word_frequencies = ck.count_word_frequencies(keywords)

print(word_frequencies)

embeddings, principal_components = ck.sif_embedding(keywords, model, word_frequencies, n_principal_components=1,
    alpha=1e-3, principal_components=None, return_components=True)

print("embeddings[5,:]:\n", embeddings[:,:5])
print("principal_components[5,:]\n", principal_components[:,:5])

embedder = ck.SIFEmbedder(model)

try:
    embedder.embed(keywords)
except RuntimeError:
    print("Embedder exception ok.")

embeddings1 = embedder.fit_embed(keywords)

print("Embeddings same:", np.array_equal(embeddings, embeddings1))
print("Principal components same:", np.array_equal(principal_components, embedder.principal_components))

param_json = embedder.serialize()

print("serialization done")

# print("serialized parameters:\n", param_json)

embedder2 = ck.SIFEmbedder(model)
embedder2.load(param_json)

print("Compare to deserialized:")
print("    Word frequencies same:", embedder.word_frequencies == embedder2.word_frequencies)
print("    Principal components same:", np.array_equal(embedder.principal_components, embedder2.principal_components))
print("    Embeddings same:", np.array_equal(embeddings, embedder2.embed(keywords)))


de_embedder_parameters_json = open("/home/ew-shopp/fasttext/de_embedder.json").read()
de_embedder = ck.SIFEmbedder(model)
de_embedder.load(de_embedder_parameters_json)

print("Loaded DE embedder!")

words = np.load(open('/home/ew-shopp/fasttext/keywords_GER_used.npy', 'rb'))
embs = np.load(open('/home/ew-shopp/fasttext/keywords_GER_emb.npy', 'rb'))

res = ck.find_closest(["fussbal", "kleidung"],  words, embs, de_embedder)
for x in res[0]:
    print(x[0], x[1])
print("\n")
for x in res[1]:
    print(x[0], x[1])

print("=" * 80)

# get categories
category_rows = []
with open('/home/ew-shopp/fasttext/productsservices_GER.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Category table column names are {", ".join(row)}')
        else:
            category_rows.append(row)
        line_count += 1
    print(f'Loaded {line_count} categories.')
# clean categories
# categories = [row[2].replace("/", " ").replace("&", " ") for row in category_rows]
# categories = [re.sub(" +", " ", category.strip().lower()) for category in categories]
categories = [row[2] for row in category_rows]
for x in categories[:10]: print(x)
category_ids = [row[0] for row in category_rows]

# build categorizer
categorizer = ck.Categorizer(de_embedder)
print("categorizer built!")

categorizer.fit(categories, category_ids=category_ids)
print("categorizer fitted!")

kws = ["fussbal schuhe", "iPhone X", "iphone", "Autoreifen", "winterreifen", "laptop keyboard", "hamburger", "pizza", "milch"]
kws_cats = categorizer.categorize(kws)
for kw, cats in zip(kws, kws_cats):
    print(kw, ":", cats, "\n")

# fussbal schuhe : [('bekleidung schuhwerk turnschuhe cross-training schuhe', 0.33673640566784835), ('bekleidung schuhwerk lässige schuhe holzschuhe', 0.39621237672523246), ('bekleidung schuhwerk turnschuhe skate-schuhe', 0.4050881830369575)]

# iPhone X : [('finanzen investieren altersvorsorgeinvestitionen 401 (k) s', 0.9193326414918438), ('jobs ausbildung jobs karriere schreiben sie weiter resume anschreiben - beispiele und vorlagen', 0.9239172365020019), ('computer unterhaltungselektronik computers software software-entwicklung programmier- und entwickler-software c c ++ software', 0.9610005672943072)]

# Autoreifen : [('lebensmittel lebensmittel haushaltswaren papiertücher und haushaltspapierprodukte kosmetiktücher', 0.6481680578740041), ('lebensmittel lebensmittel haushaltswaren lebensmittelverpackungen lagerung von lebensmitteln plastikfolie folie', 0.6556375234811832), ('lebensmittel lebensmittel haushaltswaren papiertücher und haushaltspapierprodukte', 0.658233148059344)]
