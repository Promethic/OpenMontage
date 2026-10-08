# Studio provider execution contract

Contract version 1 uses a service-owned JSON-lines worker. Studio retains the
transactional film budget, operation identity and result history. OpenMontage
contains gateway metadata, the scoped client, the HTTP adapter and the private
tool executor. The worker is not a second server or a creative orchestrator.

For each provider write the worker emits a `sending` stage and waits for the
parent's durable acknowledgement. It emits the response's order ID and queue
URLs and waits again before returning control to the SDK. Retries of an identical
write are rejected, fal server retries are disabled, and no gateway fallback is
permitted. A missing response is not evidence of a free operation.

The pinned catalog currently contains 35 paid tools and 37 gateway routes,
including the paid branches of the deprecated `image_gen` tool. Free stock APIs
and selector tools are not separately billed. New API tools without reviewed
billing metadata fail closed. Cloud tools whose old estimator returned zero
still require a verified Studio price envelope.

## Standalone connection

Obtain a scoped Studio connection using the existing Studio MCP/session setup.
Set `STUDIO_API_URL`, `STUDIO_API_TOKEN`, and `STUDIO_PROJECT_ID` in the client
process, not provider API keys. Projects must be registered in Studio. Pass
`studio_route_id`, `studio_purpose`, `scene_id`, and `studio_idempotency_key` to
paid tools. Local input references must identify media owned by that project.
Studio controls output locations; caller-supplied output paths are not authority.

The client returns the Studio operation ID on pending results. Reopen that
operation in Studio to refresh it. A deliberate new take uses a new key only
after the preceding operation is resolved. CostTracker delegates paid entries
to Studio and must not release or independently settle its reservations.

## Pricing

Studio requires an operator-reviewed price cap, source, verification date and
parameter schema for generic routes. The schema bounds model choices, quantity,
duration, input size and paid options. Exact-input schemas are useful for custom
Modal services or providers without a reliable reusable formula. Expired or
absent prices block execution rather than borrowing an approximate tool estimate.

Each route declares its recovery/cancellation capabilities. fal and Replicate
can observe recorded orders. Other routes retain synchronous results and stage
IDs; an interrupted operation without a result needs provider evidence or an
operator reconciliation. Cancellation never proves a refund.
