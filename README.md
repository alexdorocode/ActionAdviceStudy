# ActionAdviceStudy
This project adapts the Introspective Action Advising (IAA) algorithm for a DQN agent, applying it to transfer knowledge from a simpler to a more complex NES Super Mario level. By evaluating task similarity's effect on IAA’s effectiveness, we lay groundwork for future multi-agent research on adaptive action advisory systems.

Conclusions

This project has explored various Action Advising techniques within the field of Reinforcement Learning, analyzing their performance under different experimental conditions. This section summarizes the main results obtained and outlines potential future research directions.

Analysis of Project Results

Firstly, we examined the impact of preliminary parameter tuning, demonstrating a significant performance boost for the RGB NA and SS NA techniques. The Fine-tune technique (RGB FT) achieved comparable performance to SS NA without improvement, thereby validating the initial hypothesis.

Secondly, we assessed the use of dropout, confirming that it is unsuitable for environments requiring overfitting to solve specific tasks, as it degrades model performance. Regarding the learning rate, despite an anticipated negative impact from a higher learning rate, the difference between rates of 0.1 and 0.4 was minimal across most metrics, with some advantages observed at a rate of 0.4.

Finally, evaluating Action Advising techniques in a context where the distance between source and target tasks is minimal did not yield optimal results, contradicting the hypothesis that these techniques would perform better in this scenario. Additionally, the TSUA technique, being the most restrictive, showed lower performance, challenging the hypothesis that more restrictive conditions would enhance performance. In examining the parameter δ, we observed that both SUA and TUA had similar performance across δ values, although SUA showed a notable improvement at δ = 0.15, partially refuting the initial hypothesis.

In relation to the implementation of the Virtual Environment, it has enabled us to meet our objectives. We conducted the proposed research successfully, developing a flexible and adaptive system. This system not only fulfills the initial requirements but also paves the way for future extensions and improvements.

The implemented system allows easy integration of new agent configurations, such as AgentPPO or AgentA2C. Adding these new configurations only requires creating new agent classes and specific training functions for each algorithm. For instance, to integrate the PPO algorithm, it would simply involve adding an AgentPPO class and a TrainPPO function that accounts for the unique aspects of this algorithm's policy updates.
