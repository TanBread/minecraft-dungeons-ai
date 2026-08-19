use candle_core::{Tensor, Result, Module};
use candle_nn::{Conv2d, VarBuilder, Conv2dConfig};

pub struct CNNFeatureExtractor {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
}

impl CNNFeatureExtractor {
    pub fn new(vs: VarBuilder, in_channels: usize) -> Result<Self> {
        let conv1_w = vs.pp("conv1").get((64, in_channels, 5, 5), "weight")?;
        let conv1_b = vs.pp("conv1").get(64, "bias")?;
        let conv1 = Conv2d::new(conv1_w, Some(conv1_b), Conv2dConfig { stride: 3, padding: 1, dilation: 1, groups: 1, ..Default::default() });

        let conv2_w = vs.pp("conv2").get((128, 64, 3, 3), "weight")?;
        let conv2_b = vs.pp("conv2").get(128, "bias")?;
        let conv2 = Conv2d::new(conv2_w, Some(conv2_b), Conv2dConfig { stride: 2, padding: 1, dilation: 1, groups: 1, ..Default::default() });

        let conv3_w = vs.pp("conv3").get((256, 128, 3, 3), "weight")?;
        let conv3_b = vs.pp("conv3").get(256, "bias")?;
        let conv3 = Conv2d::new(conv3_w, Some(conv3_b), Conv2dConfig { stride: 2, padding: 1, dilation: 1, groups: 1, ..Default::default() });

        let conv4_w = vs.pp("conv4").get((512, 256, 3, 3), "weight")?;
        let conv4_b = vs.pp("conv4").get(512, "bias")?;
        let conv4 = Conv2d::new(conv4_w, Some(conv4_b), Conv2dConfig { stride: 2, padding: 1, dilation: 1, groups: 1, ..Default::default() });

        Ok(Self { conv1, conv2, conv3, conv4 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        let x = self.conv3.forward(&x)?.relu()?;
        let x = self.conv4.forward(&x)?.relu()?;
        x.flatten(1, x.rank() - 1)
    }

    pub fn flat_size(&self, in_channels: usize, h: usize, w: usize) -> usize {
        let h1 = (h + 2 - 5) / 3 + 1;
        let w1 = (w + 2 - 5) / 3 + 1;
        let h2 = (h1 + 2 - 3) / 2 + 1;
        let w2 = (w1 + 2 - 3) / 2 + 1;
        let h3 = (h2 + 2 - 3) / 2 + 1;
        let w3 = (w2 + 2 - 3) / 2 + 1;
        let h4 = (h3 + 2 - 3) / 2 + 1;
        let w4 = (w3 + 2 - 3) / 2 + 1;
        512 * h4 * w4
    }
}
