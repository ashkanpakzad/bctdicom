import typer
from bctdicom.util import read_file, write_file

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def thick_slicing(input_file: str, 
                  output_file: str,
                  axis: int = -1,
                  input_mm: list[float] = [1.0, 1.0, 1.0],
                  thickness_mm: float = 3,
                  spacing_mm: float = 1.5,
                  threshold: float = 2e-10,
                  MIP: bool = False):
    logger.info(f"Thick slicing {input_file} to {output_file}")
    thin_data = read_file(input_file)
    thick_data, mm, thickness_mm = thick_slicing(thin_data, axis, input_mm, thickness_mm, spacing_mm, threshold, MIP)
    write_file(output_file, thick_data, mm, thickness_mm)

if __name__ == "__main__":
    typer.run(app)