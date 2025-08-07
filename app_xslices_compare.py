from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import pyvista as pv
import plotly.express as px
import time
import os
import shutil
import uuid
import numpy as np
from dash import State
import dash

from simulation_database import simulations_database
from simulation_database import xslices_database

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "CFD Dashboard"


app.layout = html.Div(
    [
        html.H1(
            "Imperial Front Wing with Rotating Wheel CFD Analysis",
            style={"textAlign": "center"},
        ),
        html.Div("Select a CFD Simulation"),
        dcc.Dropdown(
            options=simulations_database["SimulationName"],
            value=simulations_database["SimulationName"].iloc[0],
            id="dropdown-selection-simulationame",
            persistence=True,  # ✅ enable state saving
            persistence_type="session",  # 'local', 'session', or 'memory'
        ),
        html.Div("Select a second CFD Simulation"),
        dcc.Dropdown(
            options=simulations_database["SimulationName"],
            value=simulations_database["SimulationName"].iloc[0],
            id="dropdown-selection-simulationame2",
            persistence=True,  # ✅ enable state saving
            persistence_type="session",  # 'local', 'session', or 'memory'
        ),
        html.H2("Simulation Summary"),
        html.Div(
            id="simulation-summary-content",
            style={
                "whiteSpace": "pre",
                "fontFamily": "Courier New, monospace",
                "border": "1px solid black",
                "padding": "10px",
                "width": "400px",
            },
        ),
        html.Div(
            [
                # Dropdown next to the figure
                html.Div(
                    [
                        html.Div(
                            "Select X Slice:",
                            style={"marginRight": "10px", "fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id="dropdown-xslice",
                            options=xslices_database["label"],
                            value=xslices_database["label"][0],
                            persistence=True,  # ✅ enable state saving
                            persistence_type="session",
                            style={"width": "200px"},
                        ),
                        html.Div(
                            "Select Variable:",
                            style={"marginRight": "10px", "fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id="dropdown-variable",
                            options=[
                                {"label": "Avg Cp", "value": "Avg Cp"},
                                {"label": "Avg Cp0", "value": "Avg Cp0"},
                            ],
                            value="Avg Cp0",
                            persistence=True,  # ✅ enable state saving
                            persistence_type="session",
                            style={"width": "200px"},
                        ),
                        html.Div(
                            "Select Range:",
                            style={"marginRight": "10px", "fontWeight": "bold"},
                        ),
                        dcc.RangeSlider(
                            id="slider-range",
                            min=-5,
                            max=1,
                            step=0.1,
                            value=[-1, 1],
                            persistence=True,  # ✅ enable state saving
                            persistence_type="session",
                            marks={i: str(i) for i in range(-5, 1)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(
                            "Select Isocontour:",
                            style={"marginRight": "10px", "fontWeight": "bold"},
                        ),
                        dcc.Slider(
                            id="slider-isocontour",
                            min=-3,
                            max=0,
                            step=0.1,
                            value=0,
                            persistence=True,  # ✅ enable state saving
                            persistence_type="session",
                            marks={i: str(i) for i in range(-3, 0)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(
                            "Select Z axis Range [mm]:",
                            style={"marginRight": "10px", "fontWeight": "bold"},
                        ),
                        dcc.RangeSlider(
                            id="slider-zaxis-range",
                            min=-25,
                            max=1000,
                            step=25,
                            value=[-25, 1000],
                            persistence=True,  # ✅ enable state saving
                            persistence_type="session",
                            marks={i: str(i) for i in range(-5, 1)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(
                            "Select Y axis Range [mm]:",
                            style={"marginRight": "10px", "fontWeight": "bold"},
                        ),
                        dcc.RangeSlider(
                            id="slider-yaxis-range",
                            min=-1200,
                            max=0,
                            step=25,
                            value=[-1200, 0],
                            persistence=True,  # ✅ enable state saving
                            persistence_type="session",
                            marks={i: str(i) for i in range(-5, 1)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(
                            html.Button(
                                "Update Plot",
                                id="update-button",
                                n_clicks=0,
                                style={"padding": "10px 20px"},
                            ),
                            style={"textAlign": "center", "marginTop": "20px"},
                        ),
                    ],
                    style={
                        "flex": "0 0 400px",
                        "paddingRight": "10px",
                        "border": "1px solid black",
                    },
                ),  # fixed width + margin
                # Image
                html.Div(
                    [
                        html.Img(
                            id="mesh-img", style={"maxWidth": "100%", "height": "auto"}
                        )
                    ],
                    style={
                        "flex": "1",  # take remaining space
                        "border": "1px solid black",
                        "padding": "10px",
                        "boxSizing": "border-box",
                    },
                ),
                html.Div(
                    [
                        # html.Div("Box 1", style={'flex': 1, 'border': '1px solid black', 'padding': '10px'}),
                        html.Img(
                            id="mesh-img2", style={"maxWidth": "100%", "height": "auto"}
                        )
                    ],
                    style={
                        "flex": "1",  # take remaining space
                        "border": "1px solid black",
                        "padding": "10px",
                        "boxSizing": "border-box",
                    },
                ),
            ],
            style={
                "display": "flex",
                "width": "100%",
                "padding": "10px",
                "boxSizing": "border-box",
            },
        ),
    ]
)


@callback(
    Output("simulation-summary-content", "children"),
    Input("dropdown-selection-simulationame", "value"),
)
def update_text(value):
    simulation_data = simulations_database.loc[
        simulations_database["SimulationName"] == value
    ].iloc[0]
    string = (
        "Simulation Summary\n\n"
        f"SolverNameTag : {simulation_data['SimulationNameTag']}\n"
        f"Solver : {simulation_data['Solver']}\n"
        f"Temporal : Unsteady"
    )
    return string


@callback(
    Output("mesh-img", "src"),
    Output("mesh-img2", "src"),
    Input("update-button", "n_clicks"),  # Only trigger on button click
    State("dropdown-selection-simulationame", "value"),
    State("dropdown-selection-simulationame2", "value"),
    State("dropdown-xslice", "value"),
    State("dropdown-variable", "value"),
    State("slider-range", "value"),
    State("slider-isocontour", "value"),
    State("slider-zaxis-range", "value"),
    State("slider-yaxis-range", "value"),
)
def update_figures(
    n_clicks,
    sim_name,
    sim_name2,
    xslice,
    selected_variable,
    slider_range,
    isocontour_threshold,
    zaxis_range,
    yaxis_range,
):
    if n_clicks == 0:
        # Prevent update on initial load
        raise dash.exceptions.PreventUpdate
    filename1 = readslice(
        sim_name,
        xslice,
        selected_variable,
        slider_range,
        isocontour_threshold,
        zaxis_range,
        yaxis_range,
    )
    filename2 = readslice(
        sim_name2,
        xslice,
        selected_variable,
        slider_range,
        isocontour_threshold,
        zaxis_range,
        yaxis_range,
    )
    return filename1, filename2


def readslice(
    sim_name,
    xslice,
    selected_variable,
    slider_range,
    isocontour_threshold,
    zaxis_range,
    yaxis_range,
):

    print()
    # selected_variable = 'Avg Cp0'
    # xslice = 'X_600mm'
    # isocontour_threshold = 0
    clim = slider_range
    ymin = yaxis_range[0] / 1000
    ymax = yaxis_range[1] / 1000
    zmin = zaxis_range[0] / 1000
    zmax = zaxis_range[1] / 1000

    U_inf = 12.5
    rho_inf = 1.2

    start_time = time.time()

    simulation_path = simulations_database.loc[
        simulations_database["SimulationName"] == sim_name
    ].iloc[0]["Path"]
    solver_type = simulations_database.loc[
        simulations_database["SimulationName"] == sim_name
    ].iloc[0]["Solver"]
    xslices_path = simulations_database.loc[
        simulations_database["SimulationName"] == sim_name
    ].iloc[0]["Xslices Path"]

    row_idx = xslices_database[xslices_database["label"] == xslice].index[0]

    if xslices_database[solver_type][row_idx] == "missing":
        print("File is missing")
        plotter = pv.Plotter(off_screen=True)
        plotter.add_text(
            sim_name + " " + xslice + "(file is missing)",
            position="upper_left",
            font_size=14,
            color="black",
        )

    else:
        # file = xslices_path + "\\" + xslices_database[solver_type][row_idx]
        file = os.path.join(xslices_path, xslices_database[solver_type][row_idx])
        print("Reading :", file)
        # print(file)

        # """
        dataset = pv.read(file)
        print("loaded file")

        if file.split(".")[-1] == "case":
            mesh = dataset[0]
        else:  # if vtp file
            mesh = dataset
        print(file.split(".")[-1])

        list_available_variables = list(
            mesh.point_data.keys()
        )  # access the first multiblock since there is 1 multiblock per time step (but we only have the last time step)
        print(list_available_variables)
        if selected_variable == "Avg Cp0":
            if solver_type == "CharLES":
                if "AVG<P_TOTAL<>>" in list_available_variables:
                    mesh.point_data["Avg Cp0"] = mesh.point_data["AVG<P_TOTAL<>>"] / (
                        0.5 * rho_inf * U_inf**2
                    )
                elif (
                    "AVG<U>" in list_available_variables
                    and "AVG<P>" in list_available_variables
                ):
                    u_mag = np.linalg.norm(mesh.point_data["AVG<U>"], axis=1)
                    mesh.point_data["Avg Cp0"] = (
                        mesh.point_data["AVG<P>"] + u_mag**2
                    ) / (0.5 * rho_inf * U_inf**2)
                else:
                    print("total pressure not exported for this plane")
            elif solver_type == "Fidelity DES":
                if (
                    "p_timeAverage" in list_available_variables
                    and "U_timeAverage" in list_available_variables
                ):
                    u_mag = np.linalg.norm(mesh.point_data["U_timeAverage"], axis=1)
                    mesh.point_data["Avg U_mag"] = u_mag

                    mesh.point_data["Avg Cp0"] = (
                        mesh.point_data["p_timeAverage"] + 0.5 * rho_inf * u_mag**2
                    ) / (0.5 * rho_inf * U_inf**2)
                else:
                    print("total pressure cannot be computed for this plane")
            elif solver_type == "Nektar++":
                mesh.point_data["Avg Cp0"] = mesh.point_data["Cp0"]
            else:
                print("unknown solver type")
        if selected_variable == "Avg Cp":
            if solver_type == "CharLES":
                if "AVG<P>" in list_available_variables:
                    mesh.point_data["Avg Cp"] = mesh.point_data["AVG<P>"] / (
                        0.5 * rho_inf * U_inf**2
                    )
                elif "PROJ<AVG<P>>" in list_available_variables:
                    mesh.point_data["Avg Cp"] = mesh.point_data["PROJ<AVG<P>>"] / (
                        0.5 * rho_inf * U_inf**2
                    )
                else:
                    print("pressure not exported for this plane")
            elif solver_type == "Fidelity DES":
                if "p_timeAverage" in list_available_variables:
                    mesh.point_data["Avg Cp"] = mesh.point_data["p_timeAverage"] / (
                        0.5 * rho_inf * U_inf**2
                    )
                else:
                    print("pressure not exported for this plane")
            elif solver_type == "Nektar++":
                mesh.point_data["Avg Cp"] = mesh.point_data["Cp"]
            else:
                print("unknown solver type")

        points = mesh.points
        values = mesh.point_data[selected_variable]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        # """

        end_time = time.time()
        print("Reading VTK file (execution time) :", end_time - start_time, "s")

        # print("Rendering")
        start_time = time.time()
        plotter = pv.Plotter(off_screen=True)
        # """
        print("Rendering started")
        if ymin is not None and ymax is not None:
            mesh_clip = mesh.clip(normal="y", origin=(0, ymin, 0), invert=False)
            mesh_clip = mesh_clip.clip(normal="y", origin=(0, ymax, 0), invert=True)

        # Apply clipping for z bounds
        if zmin is not None and zmax is not None:
            mesh_clip = mesh_clip.clip(normal="z", origin=(0, 0, zmin), invert=False)
            mesh_clip = mesh_clip.clip(normal="z", origin=(0, 0, zmax), invert=True)

            plotter.clear()
            plotter.add_mesh(
                mesh_clip,
                scalars=selected_variable,
                clim=clim,
                cmap="viridis",
                show_scalar_bar=True,
                n_colors=20,
                scalar_bar_args={
                    "title": selected_variable,
                    "position_x": 0.25,  # Center horizontally
                    "position_y": 0.05,  # Near the bottom
                    "width": 0.5,  # Width of the scalar bar (30% of window)
                    "height": 0.05,  # Height of the scalar bar (5% of window)
                    "vertical": False,  # Horizontal bar
                    "fmt": "%.2f",  # Format the numbers
                },
            )
            if isocontour_threshold != None:
                contours = mesh_clip.contour(
                    isosurfaces=[isocontour_threshold], scalars=selected_variable
                )
                isocontour_pts = contours.points
                if len(isocontour_pts) > 0:  # only show if the isocontour exists
                    arg_top = np.argmax(isocontour_pts[:, 2])
                    print("Location of tyre Loss (top): ", isocontour_pts[arg_top, :])
                    plotter.add_mesh(contours, color="black", line_width=5)
                    plotter.add_points(
                        isocontour_pts[arg_top, :],
                        color="red",
                        point_size=10,
                        render_points_as_spheres=True,
                    )

            arg_min = np.argmin(values)
            print("Location Min : ", points[arg_min, :])
            plotter.add_points(
                points[arg_min, :],
                color="blue",
                point_size=10,
                render_points_as_spheres=True,
            )

            # plotter.show_axes()
        # """

        plotter.add_text(
            sim_name + " " + xslice, position="upper_left", font_size=14, color="black"
        )

    # Set camera for 2D top-down view (XY plane)
    plotter.view_yz()
    plotter.reset_camera()  # Auto-zoom to fit mesh
    plotter.camera.zoom(1.0)

    end_time = time.time()
    print("Finished rendering in ", end_time - start_time, "s")

    tmp_filename = f"{sim_name}_{xslice}_{selected_variable}_tmp_{uuid.uuid4().hex}.png"
    plotter.screenshot(
        f"assets/{tmp_filename}", window_size=[1920, 1820]
    )  # save in public HTTP path

    return f"assets/{tmp_filename}"


if __name__ == "__main__":
    app.run(debug=True)
