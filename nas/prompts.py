from typing import List

from langchain.output_parsers import ResponseSchema
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class BaseNASPrompt:
    def get_response_schemas(self) -> List[ResponseSchema]:
        raise NotImplementedError

    def get_prompt_template(self) -> ChatPromptTemplate:
        raise NotImplementedError


class StandardNASPrompt(BaseNASPrompt):
    def get_response_schemas(self) -> List[ResponseSchema]:
        return [
            ResponseSchema(
                name="reasoning",
                description="Brief reasoning what parameters and architecture are better to choose based on history.",
            ),
            ResponseSchema(
                name="model_type",
                description="Type of model architecture: 'informer', 'informerstack', or 'lstm'",
            ),
            ResponseSchema(
                name="d_model", description="Dimension of model (e.g. 256, 512, 1024)"
            ),
            ResponseSchema(
                name="n_heads", description="Number of heads (e.g. 4, 8, 16)"
            ),
            ResponseSchema(
                name="e_layers",
                description="Number of encoder layers (e.g. 1, 2, 3, 4)",
            ),
            ResponseSchema(
                name="d_layers", description="Number of decoder layers (e.g. 1, 2, 3)"
            ),
            ResponseSchema(
                name="d_ff", description="Dimension of fcn (e.g. 1024, 2048)"
            ),
            ResponseSchema(
                name="factor", description="Probsparse attn factor (e.g. 3, 5)"
            ),
            ResponseSchema(
                name="learning_rate", description="Learning rate (e.g. 0.0001, 0.001)"
            ),
        ]

    def get_prompt_template(self) -> ChatPromptTemplate:
        template_string = """You are an expert in Neural Architecture Search (NAS) for Time Series Forecasting.
        Your goal is to find the best model architecture and hyperparameters to minimize the Mean Squared Error (MSE) on the validation/test set.
        
        You can choose between different model architectures:
        1. 'informer' - Transformer-based model with ProbSparse attention (good for long sequences)
        2. 'informerstack' - Stacked version of Informer (more complex, potentially better accuracy)
        3. 'lstm' - Simple LSTM model (faster training, good baseline)
        
        Here is the history of tried architectures and their resulting MSE (lower is better):
        {history}
        
        Based on this history, suggest a new architecture type and hyperparameters that might yield a better (lower) MSE.
        Explore the search space intelligently. Try different architectures to see which works best.
        
        Constraints:
        - model_type: ['informer', 'informerstack', 'lstm']
        - d_model: [128, 256, 512, 768]
        - n_heads: [4, 8, 16] (must divide d_model, mainly for informer/informerstack)
        - e_layers: [1, 2, 3, 4, 5]
        - d_layers: [1, 2, 3]
        - d_ff: [512, 1024, 2048]
        - factor: [3, 5]
        - learning_rate: [1e-5, 1e-4, 1e-3]
        
        {format_instructions}
        """
        return ChatPromptTemplate.from_template(template_string)


class ETTNASPrompt(BaseNASPrompt):
    def get_response_schemas(self) -> List[ResponseSchema]:
        return [
            ResponseSchema(
                name="reasoning",
                description="Deep analysis of previous trials. Connect the physical nature of ETT data (periodicity, load-temp relationship) to model parameters. Why did the previous best model work well? Why did others fail?",
            ),
            ResponseSchema(
                name="model_type",
                description="Architecture: 'informer' (standard), 'informerstack' (better for very long seq due to distillation), or 'lstm' (baseline).",
            ),
            ResponseSchema(
                name="d_model",
                description="Embedding dimension. Must be divisible by n_heads. (e.g. 256, 512). High d_model needs more data.",
            ),
            ResponseSchema(
                name="n_heads",
                description="Attention heads. (e.g. 4, 8, 16). Capture different periodic patterns (daily, weekly).",
            ),
            ResponseSchema(
                name="e_layers",
                description="Encoder layers. (e.g. 2, 3, 4).",
            ),
            ResponseSchema(
                name="d_layers",
                description="Decoder layers. (e.g. 1, 2). Usually fewer than encoder.",
            ),
            ResponseSchema(
                name="d_ff",
                description="FCN dimension. Typically 4 * d_model (e.g. 1024, 2048).",
            ),
            ResponseSchema(
                name="factor",
                description="ProbSparse attention factor (e.g. 3, 5). Higher factor = fewer dominant queries selected.",
            ),
            ResponseSchema(
                name="learning_rate",
                description="Learning rate. ETT data is smoother than finance, can handle 1e-4 well. Try 1e-5 for fine-tuning.",
            ),
        ]

    def get_prompt_template(self) -> ChatPromptTemplate:
        system_template = """You are a Senior AI Research Scientist specializing in Long Sequence Time-Series Forecasting (LSTF). 
You are conducting Neural Architecture Search (NAS) specifically for the **Electricity Transformer Dataset (ETT)**.

YOUR GOAL: Minimize MSE (Mean Squared Error) on the Oil Temperature (OT) target.

DOMAIN KNOWLEDGE (ETT Dataset):
1. **Physics**: The data represents power load and oil temperature. It is governed by physical laws (thermal inertia) and human activity (daily/weekly cycles). It is NOT random noise like crypto.
2. **Periodicity**: Strong daily (24h) and weekly patterns. Multi-head attention is crucial here to capture different frequencies.
3. **SOTA Context**: The 'Informer' architecture was specifically designed and benchmarked on this dataset. 'InformerStack' often performs better on long prediction horizons due to distil-convolution layers reducing memory usage and extracting dominant features.
4. **Search Strategy**: 
   - Start with configurations close to the original Informer paper (d_model=512, n_heads=8).
   - If overfitting occurs (high test MSE vs low train loss), reduce `e_layers` or `d_model`.
   - LSTM is generally weaker for long sequences but serves as a sanity check for short horizons.

CONSTRAINTS:
- `d_model` MUST be divisible by `n_heads`.
- `d_ff` is usually 4x `d_model`.
- Search Space:
    - model_type: ['informer', 'informerstack', 'lstm']
    - d_model: [128, 256, 512, 768]
    - n_heads: [4, 8, 16]
    - e_layers: [1, 2, 3, 4, 5] (InformerStack usually needs 3+)
    - d_layers: [1, 2, 3]
    - d_ff: [512, 1024, 2048]
    - factor: [3, 5]
    - learning_rate: [1e-5, 1e-4, 1e-3]

{format_instructions}
"""

        human_template = """
Current Optimization History (Oldest -> Newest):
{history}

INSTRUCTIONS:
1. **Analyze**: Look at the best MSE found so far. Which parameters are common among top performers?
2. **Hypothesize**: Based on ETT data characteristics, what architecture change could improve the result? (e.g., "Increasing n_heads might better capture the 15-minute vs hourly load patterns").
3. **Propose**: Generate the next set of hyperparameters.

- If history is empty: Propose a strong baseline (InformerStack, d_model=512, n_heads=8, e_layers=3, lr=0.0001).
- If stuck in local minima: Try a radically different Learning Rate or change `model_type`.
- If refining: Make small adjustments to `d_ff` or `factor`.

Return ONLY the JSON response.
"""

        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template),
            ]
        )
