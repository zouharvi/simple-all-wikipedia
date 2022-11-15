#!/usr/bin/env python3

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

article_a = [
    "Learning disabilities, also called learning difficulties, are conditions that make it difficult to learn and understand things in the same way others do.",
    "Some people with learning difficulties also find it hard to fit in with other people because there are many things that people must know to live in society that are not easy to learn.",
    "Learning difficulties can be things that people can learn to live with on their own, like dyslexia (a difficulty with reading) and dysgraphia (a difficulty with writing).",
    "They can also be big things that mean a person needs more help (like autism).",
    "People with learning disabilities may have average intelligence.",
    "Learning disabilities are not the same as mental illnesses.",
    "They can often deal with their difficulties by doing things in different ways.",
    "Attention-deficit hyperactivity disorder (ADHD) is not a learning disability, but it may affect how a person learns."
]

article_b = [
    "Learning disability, learning disorder, or learning difficulty (British English) is a condition in the brain that causes difficulties comprehending or processing information and can be caused by several different factors.",
    'Given the "difficulty learning in a typical manner", this does not exclude the ability to learn in a different manner.',
    'Therefore, some people can be more accurately described as having a "learning difference", thus avoiding any misconception of being disabled with a lack of ability to learn and possible negative stereotyping.',
    'In the United Kingdom, the term "learning disability" generally refers to an intellectual disability, while difficulties such as dyslexia and dyspraxia are usually referred to as "learning difficulties".',
    'While learning disability and learning disorder are often used interchangeably, they differ in many ways.',
    "Disorder refers to significant learning problems in an academic area.",
    'These problems, however, are not enough to warrant an official diagnosis.',
    'Learning disability, on the other hand, is an official clinical diagnosis, whereby the individual meets certain criteria, as determined by a professional (such as a psychologist, psychiatrist, speech language pathologist, or pediatrician).',
    'The difference is in degree, frequency, and intensity of reported symptoms and problems, and thus the two should not be confused.',
    'When the term "learning disorder" is used, it describes a group of disorders characterized by inadequate development of specific academic, language, and speech skills.',
    'Types of learning disorders include reading (dyslexia), arithmetic (dyscalculia) and writing (dysgraphia).',
    "The unknown factor is the disorder that affects the brain's ability to receive and process information.",
    "This disorder can make it problematic for a person to learn as quickly or in the same way as someone who is not affected by a learning disability.",
    "People with a learning disability have trouble performing specific types of skills or completing tasks if left to figure things out by themselves or if taught in conventional ways.",
    "Individuals with learning disabilities can face unique challenges that are often pervasive throughout the lifespan.",
    "Depending on the type and severity of the disability, interventions, and current technologies may be used to help the individual learn strategies that will foster future success.",
    "Some interventions can be quite simplistic, while others are intricate and complex.",
    "Current technologies may require student training to be effective classroom supports.",
    "Teachers, parents, and schools can create plans together that tailor intervention and accommodations to aid the individuals in successfully becoming independent learners.",
    "A multi-disciplinary team frequently helps to design the intervention and to coordinate the execution of the intervention with teachers and parents.",
    "This team frequently includes school psychologists, special educators, speech therapists (pathologists), occupational therapists, psychologists, ESL teachers, literacy coaches, and/or reading specialists."
]


def normalize(sent):
    return " ".join(nltk.word_tokenize(sent.lower()))

def similarity(a, b):
    a = a.toarray()
    b = b.toarray()
    return np.dot(a, b.T)[0][0]

article_a = [normalize(l) for l in article_a]
article_b = [normalize(l) for l in article_b]


vectorizer = TfidfVectorizer(max_features=100)
vectorizer.fit(article_a + article_b)
article_a_v = vectorizer.transform(article_a)
article_b_v = vectorizer.transform(article_b)

for a_i, (a_t, a_v) in enumerate(zip(article_a, article_a_v)):
    similarities = [similarity(a_v, b_v) for b_v in article_b_v]
    closest_b_i = np.argmax(similarities)
    print(a_t)
    print(article_b[closest_b_i])
    print(np.max(similarities))
    print()