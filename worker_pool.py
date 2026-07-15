import openai
import asyncio
import tiktoken
import traceback


class WorkerPool:
    def __init__(self, client: openai.AsyncOpenAI, model, num_workers: int):
        self.client = client
        self.model = model
        self.num_workers = num_workers
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')

        print(f"Initialized WorkerPool with concurrency {num_workers}")

    async def get_chat_completions(self, chat_completion_requests: dict[int, str], response_format):
        # Bound concurrency to num_workers; gather returns when all requests finish.
        sem = asyncio.Semaphore(self.num_workers)

        async def run_one(key, prompt):
            async with sem:
                return await self._classify_one(key, prompt, response_format)

        return await asyncio.gather(*(run_one(k, p) for k, p in chat_completion_requests.items()))

    async def _classify_one(self, key, prompt, response_format):
        # The OpenAI client handles retries/backoff for transient errors (rate
        # limits, timeouts, 5xx, connection errors) internally, honoring the
        # server's Retry-After header. Configure this via `max_retries` when
        # constructing the client (see classify.py).
        try:
            # Ensure we don't make a request that's too big
            prompt_len = len(self.tokenizer.encode(prompt))
            if prompt_len > 45_000:
                raise Exception(f"Prompt too long: {prompt_len} tokens")

            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that helps people to analyze conversations."},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
                temperature=0
            )

            print(f"Processed request: {key}")
            return (key, prompt, completion.to_dict(), None)

        except Exception as exc:
            # Either a non-retryable error or one that already exhausted the
            # client's built-in retries. Record it and move on.
            print(f"ERROR: request {key} failed: {exc} of type {type(exc)}")
            return (key, prompt, None, (str(exc), traceback.format_exc()))