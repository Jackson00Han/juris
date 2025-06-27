# modules/template/v1.0.0/template.py
import os, jinja2

TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), "templates")  

def run(inputs: dict) -> dict:
    template_id = inputs["template_id"]
    data = inputs["data"]

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_ROOT))
    tpl = env.get_template(f"{template_id}.j2")
    rendered = tpl.render(**data)
    return { "generated_text": rendered }
