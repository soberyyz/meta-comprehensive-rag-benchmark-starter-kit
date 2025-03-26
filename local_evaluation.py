import argparse
import json
import os
import random
import string
from typing import Optional, Dict, List, Any, Callable
import datetime

from agents.base_agent import BaseAgent

# Set tokenizers parallelism before importing any HF libraries
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from datasets import load_dataset

import tqdm
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn
from rich.table import Table
import copy
import tqdm.rich
from p_tqdm import p_map

from agents.user_config import UserAgent

load_dotenv()
console = Console()


class EvaluationResult(BaseModel):
    """Structured output model for evaluation results"""

    accuracy: bool
    raw_response: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "EvaluationResult":
        """Create an EvaluationResult from JSON data, handling missing fields"""
        return cls(
            accuracy=json_data.get("accuracy", False),
            raw_response=json_data.get("raw", ""),
            confidence=json_data.get("confidence", None),
            reasoning=json_data.get("reasoning", None),
        )


# Constants for configuration
DEFAULT_EVAL_MODEL = "gpt-4o-mini"
MAX_API_RETRIES = 3
DEFAULT_NUM_WORKERS = 8


def get_system_message() -> str:
    """Returns the system message for the evaluator."""
    return """You are an expert evaluator for question answering systems. Your task is to determine if a prediction correctly answers a question based on the ground truth.
    
Rules:
1. The prediction is correct if it captures all the key information from the ground truth.
2. The prediction is correct even if phrased differently as long as the meaning is the same.
3. The prediction is incorrect if it contains incorrect information or is missing essential details.
4. If the user clearly states "I don't know", count it as a "miss", not a hallucination.
    
Output a JSON object with a single field 'accuracy' whose value is true or false."""


def attempt_api_call(client, model_name, messages, max_retries=MAX_API_RETRIES):
    """
    Attempt a structured API call with retries upon encountering specific errors.

    Args:
        client: The API client to use
        model_name: The model to query
        messages: List of message objects for the conversation
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary with accuracy and raw response, or None if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            # Use completion.create instead of parse to avoid using the EvaluationResult class in worker processes
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )

            # Parse the JSON content manually
            content = response.choices[0].message.content
            try:
                result_json = json.loads(content)
                accuracy = result_json.get("accuracy", False)
                # Return both the parsed result and raw JSON for debugging
                return {"accuracy": accuracy, "raw": content}
            except json.JSONDecodeError:
                console.print(
                    f"[yellow]Failed to parse JSON from response: {content}[/yellow]"
                )
                if attempt == max_retries - 1:
                    return {"accuracy": False, "raw": content}
        except Exception as e:
            console.print(
                f"[yellow]API call failed on attempt {attempt + 1}, retrying: {str(e)}[/yellow]"
            )
            if attempt == max_retries - 1:
                console.print(
                    f"[red]Failed after {max_retries} attempts: {str(e)}[/red]"
                )
    return None


def evaluate_response(
    example_data: Dict[str, Any], eval_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate a single response and return evaluation results"""
    # Extract relevant data
    agent_response = example_data["agent_response"]
    ground_truth = example_data["ground_truth"]
    query = example_data["query"]

    # Initial evaluation
    is_idk = "i don't know" in agent_response.lower()
    is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
    is_semantically_correct = False
    api_response = None

    # Determine correctness
    is_correct = is_exact_match  # Start with exact match

    # If not an exact match and we have an evaluation model, use semantic evaluation
    if not is_exact_match and eval_model_name:
        # Create a new OpenAI client inside this function
        local_openai_client = OpenAI()

        # Prepare API call - same format regardless of IDK status
        messages = [
            {"role": "system", "content": get_system_message()},
            {
                "role": "user",
                "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n",
            },
        ]

        # Make the API call
        api_response = attempt_api_call(local_openai_client, eval_model_name, messages)

        if api_response:
            is_semantically_correct = api_response["accuracy"]
            # Only update is_correct if it's not an IDK response
            if not is_idk:
                is_correct = is_semantically_correct
    elif is_exact_match:
        # Exact matches are always semantically correct
        is_semantically_correct = True

    # Return a dictionary with evaluation results
    return {
        **example_data,  # Include all original data
        "is_exact_match": is_exact_match,
        "is_correct": is_correct,
        "is_miss": is_idk,
        "is_semantically_correct": is_semantically_correct,
        "api_response": api_response,
    }


def evaluate_agent(
    dataset: Dataset,
    agent: BaseAgent,
    openai_client: Optional[Any] = None,
    eval_model_name: Optional[str] = None,
    num_examples: Optional[int] = None,
    show_progress: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Dict[str, Any]:
    """
    Evaluate an agent on a dataset and return performance metrics.

    Args:
        dataset: The dataset to evaluate on
        agent: The agent to evaluate (must implement generate_response)
        openai_client: OpenAI client for semantic evaluation (optional)
        eval_model_name: OpenAI model name for semantic evaluation (optional)
        num_examples: Maximum number of examples to evaluate (None for all)
        show_progress: Whether to display a progress bar
        num_workers: Number of parallel workers for evaluation

    Returns:
        Dictionary containing evaluation metrics and example results
    """
    # Log evaluation settings
    console.print(f"[blue]Starting evaluation with {num_workers} workers[/blue]")
    if eval_model_name:
        console.print(
            f"[blue]Using semantic evaluation with model: {eval_model_name}[/blue]"
        )

    # Initialize results structure
    results = {
        "correct_exact": 0,
        "correct": 0,
        "miss": 0,
        "total": 0,
        "examples": [],
        "metadata": {
            "eval_model": eval_model_name,
            "agent_type": agent.__class__.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_examples": num_examples,
        },
    }

    total_examples = (
        len(dataset) if num_examples is None else min(num_examples, len(dataset))
    )

    # Phase 1: Generate agent responses (keep sequential for API stability)
    all_examples = []

    # Use tqdm.rich if show_progress is True, otherwise use a regular range
    iterator = (
        tqdm.rich.tqdm(range(total_examples), desc="Generating responses")
        if show_progress
        else range(total_examples)
    )
    for i in iterator:
        example = dataset[i]
        examples_with_responses = generate_responses_for_example(example, i, agent)
        all_examples.extend(examples_with_responses)

    # Phase 2: Evaluate responses in parallel
    # Create a function that takes only the data and model name, not the client
    def parallel_evaluate(example_data):
        return evaluate_response(example_data, eval_model_name)

    # Use p_map for parallel processing with progress bar
    eval_results = p_map(
        parallel_evaluate,
        all_examples,
        desc="Evaluating responses",
        num_cpus=num_workers,
        disable=not show_progress,
    )

    # Aggregate results and calculate metrics
    aggregate_evaluation_results(results, eval_results)

    return results

def evaluate_agent_mt(
    dataset: Dataset,
    agent: BaseAgent,
    openai_client: Optional = None,
    eval_model_name: Optional[str] = None,
    num_examples: Optional[int] = None,
    show_progress: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Dict[str, Any]:
    """
    Evaluate an agent on a dataset and return performance metrics.

    Args:
        dataset: The dataset to evaluate on
        agent: The agent to evaluate (must implement generate_response)
        openai_client: OpenAI client for semantic evaluation (optional)
        eval_model_name: OpenAI model name for semantic evaluation (optional)
        num_examples: Maximum number of examples to evaluate (None for all)
        show_progress: Whether to display a progress bar
        num_workers: Number of parallel workers for evaluation

    Returns:
        Dictionary containing evaluation metrics and example results
    """
    # Log evaluation settings
    console.print(f"[blue]Starting evaluation with {num_workers} workers[/blue]")
    if eval_model_name:
        console.print(
            f"[blue]Using semantic evaluation with model: {eval_model_name}[/blue]"
        )

    total_examples = (
        len(dataset) if num_examples is None else min(num_examples, len(dataset))
    )

    # Initialize results structure
    results = {
        "correct_exact": 0,
        "correct": 0,
        "miss": 0,
        "total": 0,
        "avg_convo_score": 0,
        "examples": [],
        "metadata": {
            "eval_model": eval_model_name,
            "agent_type": agent.__class__.__name__,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_examples": total_examples,
            "num_turns": num_examples,
        },
    }

    # Phase 1: Generate agent responses (keep sequential for API stability)
    all_examples = []

    # Use tqdm.rich if show_progress is True, otherwise use a regular range
    eval_results = []
    total_convo_score = 0
    iterator = (
        tqdm.rich.tqdm(range(total_examples), desc="Generating responses")
        if show_progress
        else range(total_examples)
    )
    for i in iterator:
        n_failed_turns = 0
        example = dataset[i]
        examples_with_responses = generate_responses_for_example(example, i, agent)
        all_examples.extend(examples_with_responses)

        example_eval_result = []
        for j, turn_data in enumerate(examples_with_responses):
            if n_failed_turns == 2:
                turn_eval_result = {
                    **turn_data,
                    "is_exact_match": False,
                    "is_correct": False,
                    "is_miss": True,
                    "is_semantically_correct": False,
                    "api_response": {
                        "accuracy": False,
                        "raw": '{\n  "accuracy": false\n}',
                    },
                }
            else:
                turn_eval_result = evaluate_response(turn_data, eval_model_name)
                n_failed_turns += 1 if not turn_eval_result["is_correct"] else 0
            example_eval_result.append(turn_eval_result)
            eval_results.append(turn_eval_result)
            print("Turn eval result:")
            print(turn_eval_result)
            print()
        total_convo_score += calc_mt_metric(example_eval_result)
        print(calc_mt_metric(example_eval_result))
        print()

    # Aggregate results and calculate metrics
    aggregate_evaluation_results(results, eval_results)
    results["avg_convo_score"] = total_convo_score / total_examples

    return results

def calc_mt_metric(eval_results: List[Dict[str, Any]]) -> float:
    """Calculate the multi-turn metric for a conversation"""
    # Initialize variables
    total = 0
    correct = 0
    missing = 0
    hallucination = 0

    # Iterate over the evaluation results
    for result in eval_results:
        # Check if the result is correct
        if result["is_correct"]:
            correct += 1
        if result["is_miss"]:
            missing += 1
        else:
            hallucination += 1
        total += 1

    return (correct - hallucination) / total

def aggregate_evaluation_results(
    results: Dict[str, Any], eval_results: List[Dict[str, Any]]
) -> None:
    """
    Aggregate individual evaluation results into the final results dictionary.

    Args:
        results: The results dictionary to update
        eval_results: List of individual evaluation results
    """
    # Aggregate results
    for result in eval_results:
        results["total"] += 1
        results["correct_exact"] += result["is_exact_match"]
        results["correct"] += result["is_correct"]
        results["miss"] += result["is_miss"]
        results["examples"].append(result)

    # Calculate metrics
    n = results["total"]
    if n > 0:
        results["exact_accuracy"] = results["correct_exact"] / n
        results["accuracy"] = results["correct"] / n
        results["missing"] = results["miss"] / n
        results["hallucination"] = (n - results["correct"] - results["miss"]) / n
        results["score"] = (2 * results["correct"] + results["miss"]) / n - 1
    else:
        results["exact_accuracy"] = 0
        results["accuracy"] = 0
        results["missing"] = 0
        results["hallucination"] = 0
        results["score"] = 0


def generate_responses_for_example(
    example: Dict[str, Any], idx: int, agent: BaseAgent
) -> List[Dict[str, Any]]:
    """Generate agent responses for all turns in an example and return structured data"""
    examples_with_responses = []

    # Determine if this is single or multi-turn
    is_multi_turn = len(example["turns"]) > 1
    answer_history = []


    # Create a mapping from interaction_id to answer for easy lookup
    answer_lookup = {a["interaction_id"]: a["ans_full"] for a in example["answers"]}

    # Process each turn in the conversation
    for turn_idx, turn in enumerate(example["turns"]):
        query = turn["query"]
        image = example["image"]  # Image is at the session level

        # Get the ground truth from the answers lookup
        interaction_id = turn["interaction_id"]
        ground_truth = answer_lookup.get(interaction_id, "No answer found")

        conversation_history = []
        if turn_idx > 0:
            conversation_history = [
                example["turns"][:turn_idx],
                answer_history,
            ]

        # Generate agent response with history for multi-turn conversations
        agent_response = agent.generate_response(
            query=query, image=image, conversation_history=conversation_history
        )

        # Store example data with response for later evaluation
        examples_with_responses.append(
            {
                "example_idx": idx,
                "turn_idx": turn_idx,
                "query": query,
                "ground_truth": ground_truth,
                "agent_response": agent_response,
                "is_multi_turn": is_multi_turn,
                "history": (
                    copy.deepcopy(conversation_history) if is_multi_turn else None
                ),
            }
        )
        answer_history.append(
            {"interaction_id": interaction_id, "agent_response": agent_response}
        )

    return examples_with_responses


def display_results(eval_results: Dict[str, Any], num_examples: int = 3) -> None:
    """Display evaluation results in a formatted way"""
    console.print("\n[bold]Evaluation Results:[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    # print("eval_results:")
    # print(eval_results)
    # print()

    if eval_results["examples"][0]["is_multi_turn"]:
        table.add_row(
            "Total conversations", str(eval_results["metadata"]["num_examples"])
        )
        table.add_row("Total turns", str(eval_results["total"]))
        table.add_row("Avg conversation score", "{:.2f}".format(eval_results["avg_convo_score"]))
    else:
        table.add_row("Total examples", str(eval_results["total"]))
    table.add_row("Exact matches", str(eval_results["correct_exact"]))
    table.add_row('"I don\'t know" responses', str(eval_results["miss"]))
    table.add_row(
        "Hallucinated responses",
        str(eval_results["total"] - eval_results["correct"] - eval_results["miss"]),
    )
    table.add_row("Exact accuracy", f"{eval_results['exact_accuracy']:.2%}")
    table.add_row("Accuracy", f"{eval_results['accuracy']:.2%}")
    table.add_row("Missing rate", f"{eval_results['missing']:.2%}")
    table.add_row("Hallucination rate", f"{eval_results['hallucination']:.2%}")
    table.add_row("Score", f"{eval_results['score']:.4f}")

    console.print(table)

    if num_examples > 0:
        console.print("\n[bold cyan]Sample Evaluation Results[/bold cyan]")

        for i, example in enumerate(eval_results["examples"][:num_examples]):
            # Determine the status and style
            if example["is_exact_match"]:
                status_text = "[green]EXACT MATCH[/green]"
                status_style = "green"
            elif example["is_semantically_correct"]:
                status_text = "[blue]SEMANTICALLY CORRECT[/blue]"
                status_style = "blue"
            elif example["is_miss"]:
                status_text = "[yellow]I DON'T KNOW[/yellow]"
                status_style = "yellow"
            else:
                status_text = "[red]INCORRECT[/red]"
                status_style = "red"

            # Create better title format
            if example["is_multi_turn"]:
                # For multi-turn examples, show current turn out of total conversation
                # We can determine total turns from history length + 1 (current turn)
                total_turns = (
                    len(example["history"][0]) if example["history"] else 0
                ) + 1
                title = f"Example {i+1} - Turn {example['turn_idx'] + 1} of {total_turns} (Status: {status_text})"
            else:
                title = f"Example {i+1} (Status: {status_text})"

            # Build the combined content
            content = []

            # Add conversation history if this is a multi-turn example
            if (
                example["is_multi_turn"]
                and example["history"]
                and example["turn_idx"] > 0
            ):
                content.append("[bold blue]Conversation History:[/bold blue]")
                content.append("")

                prev_turns, prev_answers = example["history"]
                for t_idx, (turn, answer) in enumerate(zip(prev_turns, prev_answers)):
                    content.append(f"[cyan]User:[/cyan] {turn['query']}")
                    content.append(f"[green]Agent:[/green] {answer['agent_response']}")
                    # Add a separator between history items
                    if t_idx < len(prev_turns) - 1:
                        content.append("")

                # Separator between history and current turn
                content.append("")
                content.append("─" * 30)  # Add a divider line
                content.append("")

            # Current query
            content.append("[bold cyan]Query:[/bold cyan]")
            content.append(example["query"])
            content.append("")

            # Ground truth
            content.append("[bold green]Ground Truth:[/bold green]")
            content.append(example["ground_truth"])
            content.append("")

            # Agent response
            content.append("[bold yellow]Agent Response:[/bold yellow]")
            content.append(example["agent_response"])

            # API response if available
            if example["api_response"]:
                content.append("")
                content.append("[bold blue]API Response:[/bold blue]")
                if (
                    isinstance(example["api_response"], dict)
                    and "raw" in example["api_response"]
                ):
                    # Truncate the API response to keep it manageable
                    api_text = example["api_response"]["raw"]
                    if len(api_text) > 100:
                        api_text = api_text[:100] + "..."
                    content.append(api_text)
                else:
                    content.append(str(example["api_response"]))

            # Join all content with newlines
            full_content = "\n".join(content)

            # Create and display the panel
            panel = Panel(
                full_content,
                title=title,
                border_style=status_style,
                expand=False,
                padding=(1, 2),
            )

            console.print(panel)
            console.print("")  # Add space between examples


def save_results(eval_results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        eval_results: The evaluation results to save
        output_path: Path where to save the results
    """
    # Create a copy to avoid modifying the original
    results_to_save = copy.deepcopy(eval_results)

    # Convert API responses to strings to ensure JSON serializability
    for example in results_to_save["examples"]:
        if example["api_response"]:
            example["api_response"] = str(example["api_response"])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        with open(output_path, "w") as f:
            json.dump(results_to_save, f, indent=2)
        console.print(f"[green]Results saved to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving results: {str(e)}[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an agent on the CRAG-MM dataset"
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="single-turn",
        choices=["single-turn", "multi-turn"],
        help="Dataset type to load",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="sample",
        help="Dataset split to use ('train', 'validation', 'test', 'sample')",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=3,
        help="Number of evaluation examples to show",
    )
    parser.add_argument(
        "--disable_llm_judge",
        default=False,
        action="store_true",
        help="Use semantic evaluation with OpenAI",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help="OpenAI model for semantic evaluation",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help=f"Path to save results JSON"
    )
    parser.add_argument(
        "--no_progress", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="v0.1.0",
        help="Dataset revision/version to use when loading from HuggingFace",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of worker processes for parallel evaluation (default: {DEFAULT_NUM_WORKERS})",
    )
    args = parser.parse_args()

    # Load dataset
    console.print(f"[bold blue]Loading {args.dataset_type} dataset...[/bold blue]")

    # Load dataset from either HuggingFace
    # Construct repository name for HuggingFace
    repo_name = f"crag-mm-2025/crag-mm-{args.dataset_type}-public"
    console.print(
        f"[bold green]Loading from HuggingFace:[/bold green] {repo_name} (revision: {args.revision})"
    )

    dataset = load_dataset(repo_name, revision=args.revision)

    # Get available splits and select the appropriate one
    available_splits = list(dataset.keys())
    split_to_use = args.split if args.split in available_splits else available_splits[0]

    console.print(
        f"[bold green]Using split:[/bold green] '{split_to_use}' with {len(dataset[split_to_use])} examples"
    )

    if args.disable_llm_judge:
        console.print(
            Panel(
                "[bold red]WARNING: SEMANTIC EVALUATION IS DISABLED[/bold red]\n\n"
                "No calls to LLM-as-a-Judge will be made!\n"
                "Results will rely only on exact string matching.",
                title="[bold red]ATTENTION[/bold red]",
                border_style="red",
                width=100,
                padding=(2, 5),
                expand=False,
            )
        )
    # Initialize OpenAI client if needed
    openai_client = None
    if not args.disable_llm_judge:
        console.print(
            f"[bold magenta]Using semantic evaluation with model: {args.eval_model}[/bold magenta]"
        )
        openai_client = OpenAI()

    # Run evaluation
    console.print(f"[bold yellow]Running evaluation...[/bold yellow]")
    eval_results = {}
    if args.dataset_type == "single-turn":
        eval_results = evaluate_agent(
            dataset[split_to_use],
            agent=UserAgent(),
            openai_client=openai_client if not args.disable_llm_judge else None,
            eval_model_name=args.eval_model if not args.disable_llm_judge else None,
            num_examples=args.num_eval,
            show_progress=not args.no_progress,
            num_workers=args.num_workers,
        )
    elif args.dataset_type == "multi-turn":
        eval_results = evaluate_agent_mt(
            dataset[split_to_use],
            agent=UserAgent(),
            openai_client=openai_client if not args.disable_llm_judge else None,
            eval_model_name=args.eval_model if not args.disable_llm_judge else None,
            num_examples=args.num_eval,
            show_progress=not args.no_progress,
        )

    # Display results
    display_results(eval_results, num_examples=args.show_examples)

    # Save results if requested
    if args.output_path:
        save_results(eval_results, args.output_path)
