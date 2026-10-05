"""Preserve NInfer mathematical scalar formats; store matrices in BF16/A16."""

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        if parameter.projection or name in ("text/token_embedding", "text/output_head"):
            recipe.assign(name, format="bf16", method="cast_direct", activation_policy="A16Only")
    if model.config["tie_word_embeddings"]:
        recipe.share("text/output_head", "text/token_embedding")
