# =========================================================================
# Prompt Engineering Utilities
# Corresponds to Paper Table 2: Prompting Actions for Multimodal Refinement
# =========================================================================

def get_strategy_description(action_id: int) -> str:
    """
    Returns the High-level Strategy description based on Action ID.
    Corresponds to Paper Table 2.
    """
    STRATEGY_MAP = {
        1: "a1 (Visual Volumetric Scaling): Estimate served portion relative to recipe standard by analyzing visual cues like plate size and food volume.",
        2: "a2 (Implicit Additive Inference): Infer extra caloric contributions (e.g., hidden oil or sauce) by detecting textual cues in cooking instructions.",
        3: "a3 (Culinary-Guided Adjustment): Adjust for cooking technique absorption or moisture loss (e.g., oil absorption during deep-frying).",
        4: "a4 (Ingredient-Level Rectification): Check for missing or added high-calorie items (e.g., nuts, sugar, batter) that deviate from the standard recipe.",
        5: "a5 (Factorized Decomposition): Decompose the estimation into intermediate factors (portion, oil, ingredients), then derive the final residual Delta.",
        6: "a6 (Residual Anchoring): Anchoring the prediction to the baseline. Shrink Delta -> 0 unless there is compelling multimodal evidence of deviation."
    }
    # Default fallback to Anchor strategy (a6)
    return STRATEGY_MAP.get(action_id, STRATEGY_MAP[6])

def build_prompt(row: dict, action_id: int) -> str:
    """
    Constructs the input Prompt for the MLLM.
    Injects the strategy description into the system instruction.
    """
    strategy_desc = get_strategy_description(action_id)

    prompt = (
        f"System: You are an expert dietitian. Your task is to estimate the calorie residual (Delta) of a dish.\n"
        f"Dish Name: {row['菜名']}\n"
        f"Standard Recipe Ingredients: {row['xia_recipeIngredient']}\n"
        f"Baseline Calories (USDA Anchor): {row['y_base']} kcal\n\n"
        f"Selected Reasoning Strategy: {strategy_desc}\n\n"
        f"Instruction: Based on the image and the strategy above, determine how much the actual dish's calories "
        f"deviate from the baseline. If the dish looks exactly like the standard recipe, Delta should be near 0.\n"
        f"Output strictly in JSON format:\n"
        f"{{\"delta_kcal\": <float>, \"raw_lower\": <float>, \"raw_upper\": <float>}}"
    )
    return prompt