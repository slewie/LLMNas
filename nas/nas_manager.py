import copy
from typing import Dict, Any

import torch
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from utils.logger import logger
from exp.exp_informer import Exp_Informer


class NASManager:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model_name: str = "google/gemini-2.5-flash-lite",
    ):
        self.llm = ChatOpenAI(
            api_key=api_key, base_url=base_url, model=model_name, temperature=0.7
        )

        self.history = []  # Храним истории архитектур и их метрик

        self.response_schemas = [
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
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas
        )
        self.format_instructions = self.output_parser.get_format_instructions()

    def _get_prompt_template(self) -> ChatPromptTemplate:
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

    def suggest_architecture(self) -> Dict[str, Any]:
        prompt = self._get_prompt_template()

        history_str = "No history yet."
        if self.history:
            history_str = "\n".join(
                [f"Arch: {h['arch']}, MSE: {float(h['metric'])}" for h in self.history]
            )

        messages = prompt.format_messages(
            history=history_str, format_instructions=self.format_instructions
        )

        response = self.llm.invoke(messages)

        try:
            parsed_output = self.output_parser.parse(response.content)
            logger.info(f"Предложенные параметры LLM: {parsed_output}")
            return parsed_output
        except Exception as e:
            logger.error(f"Ошибка парсинга ответа LLM: {e}")
            logger.error(f"{response.content}")
            return None

    def run_search(self, base_args, iterations=5):
        """
        Запуск поиска архитектуры.
        """
        best_metric = float("inf")
        best_arch = None

        for i in range(iterations):
            logger.info(f"\n=== NAS {i + 1}/{iterations} ===")

            suggested_params = self.suggest_architecture()
            if not suggested_params:
                continue

            current_args = copy_args(base_args)

            try:
                model_type = suggested_params.get("model_type", "informer").lower()
                if model_type not in ["informer", "informerstack", "lstm"]:  # TODO: заменить на одну общую переменную
                    logger.warning(f"Неизвестный тип модели: {model_type}. Используем informer.")
                    model_type = "informer"
                current_args.model = model_type

                current_args.d_model = int(suggested_params["d_model"])
                current_args.n_heads = int(suggested_params["n_heads"])

                if current_args.d_model % current_args.n_heads != 0:
                    logger.warning(
                        f"d_model ({current_args.d_model}) не делится на n_heads ({current_args.n_heads}). Пропускаем."
                    )
                    continue

                current_args.e_layers = int(suggested_params["e_layers"])
                current_args.d_layers = int(suggested_params["d_layers"])
                current_args.d_ff = int(suggested_params["d_ff"])
                current_args.factor = int(suggested_params["factor"])
                current_args.learning_rate = float(suggested_params["learning_rate"])
            except ValueError as e:
                logger.error(f"Ошибка конвертации параметров: {e}")
                continue

            logger.info(f"Тестируем параметры: {suggested_params}")

            try:
                exp = Exp_Informer(current_args)

                setting = "nas_iter_{}".format(i)
                exp.train(setting)

                vali_data, vali_loader = exp._get_data(flag="val")
                criterion = exp._select_criterion()
                mse = exp.vali(vali_data, vali_loader, criterion)

                logger.info(f"MSE: {mse}")

                self.history.append({"arch": suggested_params, "metric": float(mse)})

                if mse < best_metric:
                    best_metric = mse
                    best_arch = suggested_params
                    logger.info(f"Обновляем лучшийMSE: {best_metric}")

                del exp
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Ошибка эксперимента: {e}")
                import traceback

                traceback.print_exc()

        logger.info("\n=== NAS Выполнен ===")
        logger.info(f"Лучшая архитектура: {best_arch}")
        logger.info(f"Лучший MSE: {best_metric}")
        return best_arch, best_metric


def copy_args(args):
    return copy.deepcopy(args)
