import mujoco
import musclemimic_models as mm

import mjswan

myofullbody_path = mm.get_xml_path("myofullbody")
myofullbody_mjspec = mujoco.MjSpec.from_file(str(myofullbody_path))

builder = mjswan.Builder()

mm_project = builder.add_project(name="MuscleMimic Demo")

mfb_scene = mm_project.add_scene(
    spec=myofullbody_mjspec,
    name="MyoFullBody",
)

app = builder.build()
app.launch()
