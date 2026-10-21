// Define API keys in Cloudflare Worker environment variables
// Access them with: env.OPENAI_API_KEY, env.ANTHROPIC_API_KEY, etc.
export default {
	async fetch(request, env, ctx) {
		// CORS headers for preflight requests
		if (request.method === "OPTIONS") {
			return handleCORS();
		}
		// Only accept POST requests
		if (request.method !== "POST") {
			return new Response("Method not allowed", { status: 405 });
		}

		try {
			const requestData = await request.json();
			const { provider, model, messages } = requestData;

			// Validate required parameters
			if (!provider || !model || !Array.isArray(messages) || messages.length === 0) {
				return new Response(
					JSON.stringify({ error: "Missing or invalid parameters" }),
					{ status: 400, headers: corsHeaders() }
				);
			}

			const providers = {
				openai: handleOpenAI,
				anthropic: handleAnthropic,
				google: handleGoogle,
				mistral: handleMistral,
				cohere: handleCohere,
				meta: handleMeta,
			};

			const handler = providers[provider.toLowerCase()];

			if (!handler) {
				return new Response(
					JSON.stringify({ error: "Unsupported provider" }),
					{ status: 400, headers: corsHeaders() }
				);
			}

			return await handler(env, model, messages);

		} catch (error) {
			console.error("Error processing request:", error);
			return new Response(
				JSON.stringify({ error: "Internal server error" }),
				{ status: 500, headers: corsHeaders() }
			);
		}
	},
};

// CORS handling function
function handleCORS() {
	return new Response(null, {
		status: 204,
		headers: {
			'Access-Control-Allow-Origin': '*',
			'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
			'Access-Control-Allow-Headers': 'Content-Type'
		}
	});
}

// Common CORS headers
function corsHeaders() {
	return {
		'Access-Control-Allow-Origin': '*',
		'Access-Control-Allow-Methods': 'POST, OPTIONS',
		'Access-Control-Allow-Headers': 'Content-Type'
	};
}

// OpenAI handler and message formatter functions
async function handleOpenAI(env, model, messages) {
	const apiKey = env.OPENAI_API_KEY;
	if (!apiKey) return new Response("Missing API key", { status: 500 });

	try {
		const response = await fetch("https://api.openai.com/v1/chat/completions", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Authorization": `Bearer ${apiKey}`
			},
			body: JSON.stringify({
				model,
				messages: formatOpenAIMessages(messages),
				temperature: 0.7,
				max_tokens: 2000
			})
		});

		const data = await response.json();
		if (!response.ok) {
			return new Response(
				JSON.stringify({ error: data.error?.message || "OpenAI request failed" }),
				{ status: response.status, headers: corsHeaders() }
			);
		}

		return new Response(JSON.stringify({ message: data.choices[0].message.content }), { status: 200, headers: corsHeaders() });

	} catch (error) {
		console.error("Error calling OpenAI API:", error);
		return new Response(JSON.stringify({ error: "OpenAI request failed" }), { status: 500, headers: corsHeaders() });
	}
}

function formatOpenAIMessages(messages) {
	return messages.map(msg => ({
		role: msg.role,
		content: msg.content
	}));
}

// Anthropic handler and message formatter functions
async function handleAnthropic(env, model, messages) {
	const apiKey = env.ANTHROPIC_API_KEY;
	if (!apiKey) return new Response("Missing API key", { status: 500 });

	try {
		const response = await fetch("https://api.anthropic.com/v1/messages", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"x-api-key": apiKey,
				"anthropic-version": "2026-10-21"
			},
			body: JSON.stringify({
				model,
				messages: formatAnthropicMessages(messages),
				max_tokens: 2000
			})
		});

		const data = await response.json();
		if (!response.ok) {
			return new Response(
				JSON.stringify({ error: data.error?.message || "Anthropic request failed" }),
				{ status: response.status, headers: corsHeaders() }
			);
		}

		return new Response(JSON.stringify({ message: data.content[0].text }), { status: 200, headers: corsHeaders() });

	} catch (error) {
		console.error("Error calling Anthropic API:", error);
		return new Response(JSON.stringify({ error: "Anthropic request failed" }), { status: 500, headers: corsHeaders() });
	}
}

function formatAnthropicMessages(messages) {
	return messages.filter(msg => msg.role !== "system").map(msg => ({
		role: msg.role === "assistant" ? "assistant" : "user",
		content: msg.content
	}));
}

// Google (Gemini) handler and message formatter functions
async function handleGoogle(env, model, messages) {
	const apiKey = env.GOOGLE_API_KEY;
	if (!apiKey) return new Response("Missing API key", { status: 500 });

	try {
		const modelPath = model.replace(/\./g, "-"); // Format model name for API path
		const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${modelPath}:generateContent?key=${apiKey}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ contents: formatGoogleMessages(messages) })
		});

		const data = await response.json();
		if (!response.ok) {
			return new Response(
				JSON.stringify({ error: data.error?.message || "Google request failed" }),
				{ status: response.status, headers: corsHeaders() }
			);
		}

		return new Response(JSON.stringify({ message: data.candidates[0].content.parts[0].text }), { status: 200, headers: corsHeaders() });

	} catch (error) {
		console.error("Error calling Google API:", error);
		return new Response(JSON.stringify({ error: "Google request failed" }), { status: 500, headers: corsHeaders() });
	}
}

function formatGoogleMessages(messages) {
	const systemMessage = messages.find(msg => msg.role === "system");
	return [
		{
			role: "user",
			parts: [{
				text: (systemMessage ? `${systemMessage.content}\n\n` : "") +
					messages.filter(msg => msg.role !== "system")
						.map(msg => `${msg.role.toUpperCase()}: ${msg.content}`)
						.join("\n\n")
			}]
		}
	];
}

// Mistral AI handler and message formatter functions
async function handleMistral(env, model, messages) {
	const apiKey = env.MISTRAL_API_KEY;
	if (!apiKey) return new Response("Missing API key", { status: 500 });

	try {
		const response = await fetch("https://api.mistral.ai/v1/chat/completions", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Authorization": `Bearer ${apiKey}`
			},
			body: JSON.stringify({
				model,
				messages: formatMistralMessages(messages),
				temperature: 0.7,
				max_tokens: 2000
			})
		});

		const data = await response.json();
		if (!response.ok) {
			return new Response(
				JSON.stringify({ error: data.error?.message || "Mistral request failed" }),
				{ status: response.status, headers: corsHeaders() }
			);
		}

		return new Response(JSON.stringify({ message: data.choices[0].message.content }), { status: 200, headers: corsHeaders() });

	} catch (error) {
		console.error("Error calling Mistral AI API:", error);
		return new Response(JSON.stringify({ error: "Mistral request failed" }), { status: 500, headers: corsHeaders() });
	}
}

function formatMistralMessages(messages) {
	return messages.map(msg => ({
		role: msg.role,
		content: msg.content
	}));
}

// Cohere handler and message formatter functions
async function handleCohere(env, model, messages) {
	const apiKey = env.COHERE_API_KEY;
	if (!apiKey) return new Response("Missing API key", { status: 500 });

	try {
		const response = await fetch("https://api.cohere.ai/v1/generate", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Authorization": `Bearer ${apiKey}`
			},
			body: JSON.stringify({
				model,
				prompt: formatCoherePrompt(messages),
				max_tokens: 2000
			})
		});

		const data = await response.json();
		if (!response.ok) {
			return new Response(
				JSON.stringify({ error: data.message || "Cohere request failed" }),
				{ status: response.status, headers: corsHeaders() }
			);
		}

		return new Response(JSON.stringify({ message: data.generations[0].text }), { status: 200, headers: corsHeaders() });

	} catch (error) {
		console.error("Error calling Cohere API:", error);
		return new Response(JSON.stringify({ error: "Cohere request failed" }), { status: 500, headers: corsHeaders() });
	}
}

function formatCoherePrompt(messages) {
	return messages.map(msg => `${msg.role.toUpperCase()}: ${msg.content}`).join("\n\n");
}

// Meta AI (Llama) handler and message formatter functions
async function handleMeta(env, model, messages) {
	const apiKey = env.META_API_KEY;
	if (!apiKey) return new Response("Missing API key", { status: 500 });

	try {
		const response = await fetch("https://api.meta.com/v1/complete", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"Authorization": `Bearer ${apiKey}`
			},
			body: JSON.stringify({
				model,
				messages: formatMetaMessages(messages),
				max_tokens: 2000
			})
		});

		const data = await response.json();
		if (!response.ok) {
			return new Response(
				JSON.stringify({ error: data.error?.message || "Meta request failed" }),
				{ status: response.status, headers: corsHeaders() }
			);
		}

		return new Response(JSON.stringify({ message: data.choices[0].message.content }), { status: 200, headers: corsHeaders() });

	} catch (error) {
		console.error("Error calling Meta API:", error);
		return new Response(JSON.stringify({ error: "Meta request failed" }), { status: 500, headers: corsHeaders() });
	}
}

function formatMetaMessages(messages) {
	return messages.map(msg => ({
		role: msg.role,
		content: msg.content
	}));
}
