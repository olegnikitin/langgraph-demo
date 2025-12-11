import readline from "readline";
import { ChatOpenAI } from "@langchain/openai";
import "dotenv/config";
import { Annotation, Command, END, interrupt, MemorySaver, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4o-mini',
  temperature: 0.5,
});

const AgentAnnotation = Annotation.Root({
  // Defines a default channel to store the messages
  ...MessagesAnnotation.spec,

  dietType: Annotation<string>(),
  fitnessLevel: Annotation<string>(),
  askCount: Annotation<number>({
    reducer: (_acc, value) => value,
    default: () => 0,
  }),
});

// This enum defines the names of the nodes for the graph
enum Nodes {
  GenerateDiet = 'generate_diet',
  AskForPreferences = 'ask_for_preferences',
  ExtractPreferences = 'extract_preferences',
  ReviewByHuman = 'review_by_human',
}

// Maximum number of times the agent asks the user to provide the diet preferences.
// After this threshold is reached, the human-in-the-loop flow starts.
const MAXIMUM_TIMES_TO_ASK_FOR_PREFERENCES = 2;

// Function that defines a node to extract the user preferences from the last user's message
export const extractPreferences = async ({ messages }: typeof AgentAnnotation.State) => {
  const lastUserMessage = messages.findLast((message) => message instanceof HumanMessage)?.content.toString() ?? '';

  const extractMessages = [
    new SystemMessage('You are a helpful AI fitness assistant. Your goal is to provide a user with a simple diet plan based on user\'s preferences.'),
    new HumanMessage(`Extract the 'diet type' and 'fitness level' from the following message. Return a stringified JSON object as a plain text without any formatting in a following format: { "dietType": <diet type>, "fitnessLevel": <fitness level> }. If no data specified for a specific parameter, set its value to null. Message: ${lastUserMessage}`)
  ];

  try {
    const result = await model.invoke(extractMessages);

    const lastResponse = result.content.toString() ?? '{}';
    const preferences = JSON.parse(lastResponse);

    return {
      messages: [...extractMessages, result],
      dietType: preferences.dietType,
      fitnessLevel: preferences.fitnessLevel,
    };
  } catch (err) {
    console.error(`Error parsing a JSON returned from AI agent: ${(err as Error).message}`);
    return {
      messages: [extractMessages],
    };
  }
};

// Function that defines a node to ask for the user diet preferences
export const askForPreferences = async ({ askCount }: typeof AgentAnnotation.State)  => {
  const followupMessage = new AIMessage('To create your diet plan, I need additional parameters such as your diet preference and fitness level.');
  console.log('* Agent:', followupMessage.content);
  const userResponse = await askQuestion();

  return {
    messages: [followupMessage, new HumanMessage(userResponse)],
    askCount: askCount + 1,
  };
};

// Fucntion to create a node to generate a diet plan
export const generateDiet = async ({ dietType, fitnessLevel }: typeof AgentAnnotation.State)  => {
  const generateMessage = new HumanMessage(`Please create a detailed diet plan of the ${dietType} diet type for a user of ${fitnessLevel} fitness level`);

  const result = await model.invoke([generateMessage]);
  return { 
    messages: [generateMessage, result], 
  };
};

// Function to create a conditional edge to check if user already provided the preferences
export const askForPreferencesOrGenerateDiet = ({ dietType, fitnessLevel, askCount }: typeof AgentAnnotation.State) => {
  if (dietType && fitnessLevel) {
    return Nodes.GenerateDiet;
  }

  if (askCount < MAXIMUM_TIMES_TO_ASK_FOR_PREFERENCES) {
    return Nodes.AskForPreferences;
  }

  return Nodes.ReviewByHuman;
};

// Function that defines a node to review the current user's preferences by the human agent.
// The human agent should be prompted to approve or reject a random diet generation by the AI agent.
export const reviewByHuman = ({ dietType, fitnessLevel }: typeof AgentAnnotation.State) => {
  const humanApproval = interrupt<string, string>(`The user didn't provide enough data to generate a diet plan, the current preferences are: dietType = "${dietType}", fitnessLevel="${fitnessLevel}". Should I generate a random diet plan? If not, I will ask the user for the preferences one more time. (Y/N/Yes/No)`);
  
  if (humanApproval.toLowerCase() === 'y' || humanApproval.toLowerCase() === 'yes') {
    return new Command({
      goto: Nodes.GenerateDiet,
      update: { dietType: 'random', fitnessLevel: 'unknown' },
    });
  }

  return new Command({
    goto: Nodes.AskForPreferences,
    update: {
      dietType, 
      fitnessLevel, 
      askCount: 0, 
    },
  });
};

const workflow = new StateGraph(AgentAnnotation)
  .addNode(Nodes.GenerateDiet, generateDiet)
  .addNode(Nodes.AskForPreferences, askForPreferences)
  .addNode(Nodes.ExtractPreferences, extractPreferences)
  .addNode(Nodes.ReviewByHuman, reviewByHuman)
  .addEdge(START, Nodes.ExtractPreferences)
  .addEdge(Nodes.AskForPreferences, Nodes.ExtractPreferences)
  .addConditionalEdges(Nodes.ExtractPreferences, askForPreferencesOrGenerateDiet)
  .addEdge(Nodes.GenerateDiet, END);

const app = workflow.compile({
  checkpointer: new MemorySaver(),
});

// The readline interface for command-line interaction.
// This allows your program to read user input from the console.
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Helper function to ask a question and get user input.
// This function returns a Promise that resolves with the user's answer.
const askQuestion = (prefix = 'You'): Promise<string> => {
  return new Promise((resolve) => {
    rl.question(`> ${prefix}: `, (answer) => resolve(answer));
  });
};

const conversationConfig = { configurable: { thread_id: 'fake_id' }};

// Function that interacts with the human agent if needed
// and asks the agent to approve or reject the random diet generation proposal
export const reviewByHumanIfNeeded = async (response: typeof AgentAnnotation.State): Promise<typeof AgentAnnotation.State> => {
  const currentState = await app.getState(conversationConfig);
  const nextNode = currentState.next[0];

  if (nextNode === Nodes.ReviewByHuman) {
    const reviewTask = currentState.tasks.find(({ name }) => name === Nodes.ReviewByHuman);
    if (!reviewTask) {
      throw new Error('Reached human review node, but the task with interrupts is missing');
    }

    const [ interrupt ] = reviewTask.interrupts;
    console.log('* Agent:', interrupt.value);

    const approval = await askQuestion('Approver');
    const nextResponse = await app.invoke(new Command({ resume: approval }), conversationConfig);

    // Continue the loop
    return reviewByHumanIfNeeded(nextResponse);
  }

  return response;
};

// Function to manage the interactive conversation loop with the agent.
// This creates a continuous dialog experience for the user.
export const interactWithAgent = async () => {
  const message = await askQuestion();
  
  // Check for exit condition
  if (["exit", "quit"].includes(message.trim().toLowerCase())) {
    console.log("Thank you for using the AI Fitness Assistant. Goodbye!");
    rl.close();
    return;
  }

  let response = await app.invoke(
    { messages: [new HumanMessage(message)] },
    conversationConfig,
  );

  // Interchain the conversation to check if the human agent's assistance is needed
  response = await reviewByHumanIfNeeded(response);

  const responseMessage = response.messages.at(-1)?.content;
  // If AI agent hasn't responded with any message, exit the dialog
  if (!responseMessage) {
    throw new Error('AI agent has not returned any response');
  }

  console.log('* Agent:', responseMessage);

  await interactWithAgent();
};

if (require.main === module) {
  interactWithAgent().catch((error) => {
    console.error("An error occurred:", error);
  });
}
