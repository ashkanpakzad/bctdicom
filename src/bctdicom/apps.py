import typer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def thick_slicing(input_file: str, output_file: str):
    logger.info(f"Thick slicing {input_file} to {output_file}")



if __name__ == "__main__":
    typer.run(app)