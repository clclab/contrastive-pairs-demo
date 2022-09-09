# Measuring bias in language models is hard!
How to measure bias in language models is not trivial and still an active area of research.
First of all, what is bias? As you may have noticed, stereotypes may change across languages and cultures.
What is problematic in the USA, may not be relevant in the Netherlands---each cultural context requires its own careful evaluation.
Furthermore, defining good ways to measure it is also difficult.
For example, [Blodgett et al. (2021)](https://aclanthology.org/2021.acl-long.81/) find that typos, nonsensical examples, and other mistakes threaten the validity of CrowS-Pairs, the dataset we show above.

<img title="Results for French and English language models" alt="Bias evaluation on the enriched CrowS-pairs corpus, after collection of new sentences in French, translation to create a bilingual corpus, revision and filtering. A score of 50 indicates an absence of bias. Higher scores indicate stronger preference for biased sentences. In header, "BT" used for "BERT" due to space constraints. https://aclanthology.org/2022.acl-long.583.pdf" src="/aggregated_results_crows-pairs.PNG">
