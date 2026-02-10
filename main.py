import os
from langchain_core.messages import HumanMessage # High level framework that lets us to build AI applications
from langchain_google_genai import ChatGoogleGenerativeAI # Allows us to use Gemini in LangChain and LangGraph
from langchain.tools import tool
from dotenv import load_dotenv # Allows us to load environment variable files from script


from langgraph.prebuilt import ToolNode
from langchain.agents import create_agent # Complex framework that lets us to build AI 'agents'

load_dotenv()

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")



# External services that the agent can call and utilize
@tool
def calculator(a: float, b: float) -> str:
    """Use basic arithmetic addition to return answers"""
    return f"The sum of {a} and {b} is {a+b}"


def main():

    # Check if API key is configured
    if not API_KEY:
        print("⚠️  Configuration Required!")
        print("=" * 50)
        print("To use this AI Assistant, you need to:")
        print("1. Get a free API key from: https://makersuite.google.com/app/apikey")
        print("2. Create a '.env' file in this directory")
        print("3. Add: GEMINI_API_KEY=your_actual_key_here")
        print("=" * 50)

        # Run demo mode
        run_demo_mode()
        return

    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
        return
    
    # The higher the temperature, the more random the model is going to be
    model = ChatGoogleGenerativeAI(temperature=0, google_api_key=api_key, model="gemini-pro")

    # AI chatbot has access to tools, giving more features. Can be updated through 'tools'
    tools = [calculator]
    model_with_tools = model.bind_tools(tools)

    print("Welcome, I'm your assistant!")
    print("We can have a chat, or you can ask me to perform some calculations")
    print("(Type 'quit' to exit)")



    while True:
        user_input = input("\nYou: ").strip()

        if user_input == 'quit':
            break

        print("\nAssistant: ",  end="")

        try:
            # Get response from model
            response = model_with_tools.invoke([HumanMessage(content=user_input)])
            

            if hasattr(response, 'tool_calls') and response.tool_calls:

                for tool_call in response.tool_calls:
                    # Find the right tool by name
                    tool_name = tool_call['name']
                    args = tool_call['args']
            
                    # Execute the appropriate tool
                    for tool in tools:
                        if tool.name == tool_name:
                            result = tool.invoke(args)
                            print(f"[Using {tool_name}]: {result}")
                            break
            else:
                print(response.content)


        except Exception as error:
            print(f'Error: {error}')

        

if __name__ == "__main__":
    main()
