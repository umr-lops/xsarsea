#!/usr/bin/env python
import os.path

import typer

import xsarsea.windspeed

app = typer.Typer()


@app.command()
def models_to_nc(
    export_dir: str = typer.Argument(..., help="destination directory"),
    sarwing_dir: str = typer.Option(None, help="sarwing top dir luts"),
):
    if sarwing_dir is not None:
        xsarsea.windspeed.register_nc_luts(sarwing_dir)
    prefix = xsarsea.windspeed.models.LutModel._name_prefix
    for model_name, row_model in xsarsea.windspeed.available_models().iterrows():
        model = row_model.model
        if not isinstance(model, xsarsea.windspeed.models.NcLutModel):
            try:
                outfile = os.path.join(export_dir, f"{prefix}{model.name}.nc")
                model.to_netcdf(outfile)
                print(f"Wrote {outfile}")
            except Exception as e:
                print(f"Error processing {model.name} : {str(e)}")
        else:
            print(f"Skipping {model_name} ({model.path})")


@app.command()
def dummy():
    # just a dummy command for typer. To be removed when a new command will be added
    pass


if __name__ == "__main__":
    app()
