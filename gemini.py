from google import genai
from litellm import Router
import time
from dotenv import load_dotenv
import os


# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()

API_KEY = os.getenv("Gemini_API_KEY_1")

# Logical model name exposed by our router.
# Your application will always request this model.
ROUTER_MODEL_NAME = "gemini-auto"

# How long a failed/rate-limited deployment stays in cooldown.
COOLDOWN_SECONDS = 60

# Number of retries against the SAME deployment.
#
# We deliberately keep this at 0.
#
# Desired behavior:
#
# Model A -> 429 -> immediately try another model
#
NUM_RETRIES = 0


# ============================================================
# GOOGLE MODEL DISCOVERY
# ============================================================

def discover_models():
    """
    Discover models available to the ONE Google API key.

    No additional accounts or API keys are used.
    """

    client = genai.Client(api_key=API_KEY)

    print()
    print("=" * 70)
    print("DISCOVERING MODELS")
    print("=" * 70)

    models = []

    for model in client.models.list():

        model_name = model.name

        if model_name.startswith("models/"):
            model_name = model_name[len("models/"):]

        supported_actions = getattr(
            model,
            "supported_actions",
            None
        )

        print(f"\nModel: {model_name}")

        if supported_actions:
            print(
                f"Supported actions: "
                f"{supported_actions}"
            )

        models.append({
            "name": model_name,
            "supported_actions": supported_actions,
        })

    print()
    print("=" * 70)

    return models


# ============================================================
# CHECK WHETHER MODEL SUPPORTS GENERATE CONTENT
# ============================================================

def supports_generate_content(model):
    """
    Return True only for models that can perform
    normal Gemini text generation.

    We don't want embedding/image/TTS/etc. models
    entering the chat routing pool.
    """

    supported_actions = model.get(
        "supported_actions"
    )

    if supported_actions:

        normalized = {
            str(action).lower().replace("_", "")
            for action in supported_actions
        }

        if "generatecontent" in normalized:
            return True

        return False

    # Some SDK versions may not expose supported_actions.

    name = model["name"].lower()

    # Exclude obvious non-chat models.
    excluded_patterns = [
        "embedding",
        "embed",
        "imagen",
        "image",
        "tts",
        "text-to-speech",
        "speech",
        "audio",
        "veo",
        "music",
        "lyria",
        "transcribe",
        "robotics",
    ]

    for pattern in excluded_patterns:

        if pattern in name:
            return False

    # Conservative fallback for normal Gemini models.
    if "gemini" in name:
        return True

    return False


# ============================================================
# BUILD LITELLM DEPLOYMENTS
# ============================================================

def build_deployments(discovered_models):
    """
    Convert discovered Gemini models into LiteLLM deployments.

    Every deployment belongs to the SAME logical model group:

        gemini-auto

    Example:

        gemini-auto
            |
            +-- gemini/gemini-2.5-flash
            +-- gemini/gemini-2.5-flash-lite
            +-- gemini/gemini-2.0-flash
    """

    deployments = []

    seen = set()

    for model in discovered_models:

        if not supports_generate_content(model):
            continue

        model_name = model["name"]

        # Prevent duplicate models.
        if model_name in seen:
            continue

        seen.add(model_name)

        deployment = {
            "model_name": ROUTER_MODEL_NAME,

            "litellm_params": {
                "model": f"gemini/{model_name}",

                # SAME Google API key for every deployment.
                "api_key": API_KEY,

                # Do not repeatedly retry the same
                # rate-limited model.
                "num_retries": NUM_RETRIES,
            },

            "model_info": {
                "id": model_name,
            },
        }

        deployments.append(deployment)

    return deployments


# ============================================================
# PRINT ROUTING POOL
# ============================================================

def print_routing_pool(deployments):

    print()
    print("=" * 70)
    print("SINGLE-ACCOUNT GEMINI ROUTING POOL")
    print("=" * 70)

    if not deployments:
        print("No compatible Gemini generation models found.")
        return

    for index, deployment in enumerate(
        deployments,
        start=1
    ):

        model = deployment[
            "litellm_params"
        ]["model"]

        print(
            f"{index:02d}. {model}"
        )

    print()
    print(
        f"Total models: {len(deployments)}"
    )

    print("=" * 70)


# ============================================================
# CREATE ROUTER
# ============================================================

def create_router(deployments):

    if not deployments:

        raise RuntimeError(
            "No Gemini generation models are available "
            "for this API key."
        )

    router = Router(

        model_list=deployments,

        # ----------------------------------------------------
        # Routing strategy
        # ----------------------------------------------------
        #
        # Randomly distribute traffic among healthy
        # deployments.
        #
        routing_strategy="simple-shuffle",

        # ----------------------------------------------------
        # Same-model retries
        # ----------------------------------------------------

        num_retries=NUM_RETRIES,

        # ----------------------------------------------------
        # Failure handling
        # ----------------------------------------------------
        #
        # A deployment that fails is temporarily removed
        # from the routing pool.
        #

        allowed_fails=0,

        cooldown_time=COOLDOWN_SECONDS,

        # ----------------------------------------------------
        # Check deployment availability before calling it.
        # ----------------------------------------------------

        enable_pre_call_checks=True,
    )

    return router


# ============================================================
# SEND REQUEST
# ============================================================

def send_request(
    router,
    prompt
):
    """
    Send one request through the logical model group.

    We NEVER specify an individual Gemini model here.

        model="gemini-auto"

    LiteLLM chooses the actual Gemini deployment.
    """

    print()
    print("=" * 70)
    print("REQUEST")
    print("=" * 70)

    print(prompt)

    print()
    print("Routing request...")

    start = time.time()

    try:

        response = router.completion(

            model=ROUTER_MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            temperature=0.7,

        )

        elapsed = time.time() - start

        print()
        print("=" * 70)
        print("RESPONSE")
        print("=" * 70)

        print(
            response.choices[0].message.content
        )

        print()
        print(
            f"Completed in {elapsed:.2f}s"
        )

        print("=" * 70)

        return response

    except Exception as error:

        elapsed = time.time() - start

        print()
        print("=" * 70)
        print("REQUEST FAILED")
        print("=" * 70)

        print(
            f"Error: {error}"
        )

        print(
            f"Time: {elapsed:.2f}s"
        )

        print("=" * 70)

        return None


# ============================================================
# INTERACTIVE CHAT
# ============================================================

def interactive_chat(router):

    print()
    print()
    print("=" * 70)
    print("GEMINI SINGLE-ACCOUNT DYNAMIC ROUTER")
    print("=" * 70)

    print()
    print("Available logical model:")
    print()
    print(f"    {ROUTER_MODEL_NAME}")

    print()
    print("Type a prompt.")
    print("Type 'exit' to stop.")
    print("=" * 70)

    while True:

        try:

            prompt = input("\nYou: ")

        except KeyboardInterrupt:

            print()
            print("Stopping...")
            break

        except EOFError:

            print()
            print("Stopping...")
            break

        prompt = prompt.strip()

        if not prompt:
            continue

        if prompt.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print("Stopping...")
            break

        send_request(
            router,
            prompt
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GEMINI + LITELLM SINGLE-ACCOUNT ROUTER")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate API key
    # --------------------------------------------------------

    if (
        not API_KEY
        or API_KEY == "YOUR_NEW_GOOGLE_API_KEY"
    ):

        raise RuntimeError(
            "\nSet your Google API key in API_KEY first."
        )

    # --------------------------------------------------------
    # Phase 1:
    #
    # Discover models available to THIS ONE API KEY.
    # --------------------------------------------------------

    discovered_models = discover_models()

    if not discovered_models:

        raise RuntimeError(
            "Google returned no models."
        )

    # --------------------------------------------------------
    # Filter and build LiteLLM deployments.
    # --------------------------------------------------------

    deployments = build_deployments(
        discovered_models
    )

    # --------------------------------------------------------
    # Show routing pool.
    # --------------------------------------------------------

    print_routing_pool(
        deployments
    )

    # --------------------------------------------------------
    # Create LiteLLM router.
    # --------------------------------------------------------

    router = create_router(
        deployments
    )

    print()
    print(
        "Router initialized successfully."
    )

    # --------------------------------------------------------
    # Start interactive interface.
    # --------------------------------------------------------

    interactive_chat(
        router
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()