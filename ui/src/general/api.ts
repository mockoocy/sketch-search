export class ApiError extends Error {
  public readonly context: string;
  public readonly response?: Response;
  constructor(message: string, context: string, response?: Response) {
    super(message);
    this.response = response;
    this.context = context;
  }
}

type ApiFetchOptions = RequestInit & {
  url: string | URL;
  context: string;
};
type FailedApiResponse = { error: string };
type ApiResponse<TData> = TData | FailedApiResponse;

export async function apiFetch<TData>({
  url,
  context,
  ...init
}: ApiFetchOptions): Promise<TData> {
  let response: Response;
  try {
    response = await fetch(url, { ...init });
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : "Network Error",
      context,
    );
  }
  let body: ApiResponse<TData>;
  try {
    body = await response.json();
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : "Could not parse JSON Response",
      context,
      response,
    );
  }
  if (body?.error && !response.ok) {
    const failedBody = body as FailedApiResponse;
    throw new ApiError(failedBody.error, context, response);
  }
  return body as TData;
}
