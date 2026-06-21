# Slate PRD
## Why
LLMs are great at processing content, especially text. But more the content to process, noisier the outcome. In a world where attention is all you need, attention on right things matters the most. The goal for slate is to reduce the amount of content that an LLM needs to process while stil ensuring that everything that deserves attention is covered.
## Things to consider
### Possible alternate memory structures:
1. Native md style used by Claude - too much generalisation or too long files
2. Grep and surrounding text - exact text match dependency but semantic keywords can be used for multiple searches
3. RAG - closest to desired model but scales badly. Most of the content being processed ends up being too close to each other in a larger universe leading to saturation with generic content. Same would ideally happen with Grep with certain keywords.

### What kind of situations to handle:
• Reducing repeated context without losing salience
    ◦ Certain common themes do repeat often but in each case there are some additional nuance which might deserve attention.
• Being able to drill down basis how specific information is required
    ◦ In a lot of cases this is handled by say creating projects, clusters, things like decision trees. But ideally should be more free flowing like a graph while being able to decide when to stop navigating further.
• Being able to optimise where processing power is to be spent at each stage
### Where is cost:
• All LLM calls - tokens exchanged - what is pushed and what comes back
• Embedding calls - input tokens
What is stored and processed:
• While raw content should be stored as it is, it should never have to be processed.
• Pre-processed structured data and various indexes shall be processed for retrievals and consolidation.
## The core principle
At each stage, are we able to figure out where to dedicate attention to?
## North Star
The goal is to be able to achieve same outcome with much lesser tokens processed given a repository of content. Essentially recall/tokens used across the pipeline.
## What
A three stage model that breaks down what deserves attention at each stage.
### Write
The first stage at which a new piece of content is pushed. A very long piece of content has to be:
• Broken down into smaller independent pieces
• Maintain relationships between these pieces where it adds critical value
• A quick run against existing memory to see what matches against something known and identifying what exactly is new and deserves more attention.
The end goal becomes defining fragments, relationships amidst them and annotating existing background + salience.

Key functions - chunking/fragment identification, relationship mapping against chunks/fragments, retrieval, pattern identification and salience tagging.

• Entire focus has to be on identifying what in this particular content deserves attention. This in turn can be optimised by
    ◦ how to define fragments - would look like a version upgrade
    ◦ Existing knowledge base - would optimise continously
    ◦ How salience is defined - mix of a version upgrade and knowledge base scale up
    
KPI would be - % regeneration/ % tokens of original. Ideal would be 95%+ regeneration at 10% of tokens of original content.
    
PS: Assuming that input tokens are 2-3x of content (because of knowledgebase being processed which will ideally be optimised in retrieval) and is fixed and output token are being measured in KPI.
### Retrieval
For retrieval, a query itself has to be decomposed. Typically it would cover some background with some specific nuances. One interesting thing could be that a nuance might be covered in a different context and might be something that can be easily borrowed in this context. While a deep dive in background might add more context to an otherwise shallow problem.

Now the goal is to not pollute the context window by fetching in just about everything. But to focus on what might need further elaboration. Which means to find what is incomplete or might benefit from further elaboration given a query. This is again a function of two things - what matters here + what the memory can provide.

Which looks pretty similar to what is happening in write. Fragmenting, assigning some value to fragments, matching against the existing knowledge base.

Key functions: defining fragments of query, prioritising which fragments to dive deeper into, fetching relevant context for independent fragments, combining all fetched context and prioritising what to push as a result and what to drop.
• Entire focus has to be on how to enrich this query with content at hand. What would just be a repeat in knowledge base and where knowledge base would add depth. Fetching and returning only elements that add clear, relevant depth.
KPI: tokens required to achieve 95% recall/raw tokens with 100% recall. Should ideally be <10%
### Consolidation
Every write and every retrieval creates signals. What was repeated, what was new. What deserved attention, what didn't. These signals can be used to optimise the workflows over time in terms of better structure of knowledge base and may be optimising how fragments are defined.
The consolidation stage hence shall process both what was written and what was retrieved. It should lead to things like - is there a common pattern that was overlooked before, treated as salience but is actually a background? Did something which was already known change? Did we add more nuance to something? Did we return something as deserving attention but it didn't? Was there something that deserved attention but was missed?

Key functions: Learning attention spread basis reinforcement signals, percolating changes to existing data structure basis new information or feedback, ensuring sanity of data structure after making all independent changes everywhere, adding weights or scores that will affect future prioritisations.

• Entire focus has to be on ensuring that every new signal is deeply incorporated in the data structure so that all future write and retrieve focuses attention on the correct things.

KPI: Net token output post consolidation (adds + updates)/raw tokens