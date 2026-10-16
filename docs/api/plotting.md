# Plotting

Choose a plot class for the data you want to display. Each class keeps a scene
open while you add data; the methods below provide static figures and 3D views.

::: anatplot.plotting.SurfacePlot
    options:
      members: [add_layer, add_parcellation, add_contacts]

::: anatplot.plotting.ROIPlot
    options:
      members: [add_roi, add_contacts]

::: anatplot.plotting.VolumePlot
    options:
      members: [add_contacts]

## Display and export

These methods apply to `SurfacePlot`, `ROIPlot`, and `VolumePlot`.

::: anatplot.plotting.SurfacePlot.to_figure
    options:
      heading_level: 3

::: anatplot.plotting.SurfacePlot.view
    options:
      heading_level: 3

::: anatplot.plotting.SurfacePlot.export_html
    options:
      heading_level: 3

::: anatplot.plotting.SurfacePlot.close
    options:
      heading_level: 3
