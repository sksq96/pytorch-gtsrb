from io import BytesIO

import PIL
import scipy.misc
import tensorboardX as tb
from tensorboardX.summary import Summary


class TensorBoard(object):

    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def image_summary(self, tag, value, step):
        for idx, img in enumerate(value):
            summary = Summary()
            bio = BytesIO()

            if type(img) == str:
                img = PIL.Image.open(img)
            elif type(img) == PIL.Image.Image:
                pass
            else:
                img = scipy.misc.toimage(img)

            img.save(bio, format="png")
            image_summary = Summary.Image(encoded_image_string=bio.getvalue())
            summary.value.add(tag=f"{tag}/{idx}", image=image_summary)
            self.summary_writer.add_summary(summary, global_step=step)

    def scalar_summary(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)


# model_dir = f"runs/n/{1}/{time.asctime(time.localtime())}/"
# tb = TensorBoard(model_dir)
# if tb is not None:
#     tb.scalar_summary("metric/loss", loss.item(), step)
#     tb.scalar_summary("metric/val_acc", val_acc, step)
