import copy
from typing import Dict, Any

import torch
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser

from utils.logger import logger
from exp.exp_informer import Exp_Informer
from nas.prompts import StandardNASPrompt, ETTNASPrompt


class NASManager:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model_name: str = "google/gemini-2.5-flash-lite",
        prompt_type: str = "standard",
    ):
        self.llm = ChatOpenAI(
            api_key=api_key, base_url=base_url, model=model_name, temperature=0.7
        )

        self.history = []  # Храним истории архитектур и их метрик

        if prompt_type == "ett":
            self.prompt_strategy = ETTNASPrompt()
        else:
            self.prompt_strategy = StandardNASPrompt()

        self.response_schemas = self.prompt_strategy.get_response_schemas()
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas
        )
        self.format_instructions = self.output_parser.get_format_instructions()

    def _get_prompt_template(self) -> ChatPromptTemplate:
        return self.prompt_strategy.get_prompt_template()

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
                if model_type not in [
                    "informer",
                    "informerstack",
                    "lstm",
                ]:  # TODO: заменить на одну общую переменную
                    logger.warning(
                        f"Неизвестный тип модели: {model_type}. Используем informer."
                    )
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
