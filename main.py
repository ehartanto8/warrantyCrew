from self_help_agent import HomeownerHelpAgent

if __name__ == "__main__":
    question = input("Enter homeowner question: ")
    agent = HomeownerHelpAgent()
    answer = agent.run(question)

    print("\n=== Final Answer ===\n", answer)
