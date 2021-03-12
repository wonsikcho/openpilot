#pragma OPENCL EXTENSION cl_khr_fp16 : enable

const half black_level = 42.0;

const __constant half3 color_checker[24] = {
  // unrgb-ed 24 color
  (half3)(0.17371761334199545, 0.07877227489274108, 0.05283996587374834),
  (half3)(0.5598949270402196, 0.27738051696188065, 0.21083303171756615),
  (half3)(0.10422591218792276, 0.18890702549537985, 0.3288664670776134),
  (half3)(0.10577713847065086, 0.1507405343175578, 0.0508226599624431),
  (half3)(0.22748036316353412, 0.21238674764911888, 0.426361519011437),
  (half3)(0.11535734563011396, 0.5074295409770233, 0.4110453235136778),
  (half3)(0.7460409079956987, 0.20205100791039962, 0.029836587047862156),
  (half3)(0.0587451698556644, 0.10118130979651963, 0.38767682056023867),
  (half3)(0.5602199605298291, 0.08008697309470449, 0.11454730998620513),
  (half3)(0.10918261627430319, 0.0419896160557127, 0.13810900791575506),
  (half3)(0.33286472914137144, 0.497339998522337, 0.04257086491903316),
  (half3)(0.7712000293516694, 0.35782121152357166, 0.020416044239498995),
  (half3)(0.020951071763526702, 0.04753638477185718, 0.2840780226932377),
  (half3)(0.04610501951591787, 0.2920399918541361, 0.061461970434956936),
  (half3)(0.4462410299780065, 0.03635235033540395, 0.04052950795205055),
  (half3)(0.8418878493283043, 0.5742987770890899, 0.004626444878787383),
  (half3)(0.522416804529426, 0.07775884448451917, 0.28924497168659535),
  (half3)(0.0, 0.23367176915049423, 0.37712141345769556),
  (half3)(0.8797385513759296, 0.8849277224514172, 0.834060437159572),
  (half3)(0.5846320156705019, 0.5921024386522283, 0.584348882651198),
  (half3)(0.35777860542167317, 0.36703687155998327, 0.3651856217692793),
  (half3)(0.1901387612558146, 0.19084481905652256, 0.18977147555248342),
  (half3)(0.08594976624328697, 0.0887256754458143, 0.08978424657188341),
  (half3)(0.0313615911870714, 0.03149234706137136, 0.03231547067341735),
};

const __constant half3 color_correction[3] = {
  // post wb CCM
  (half3)(1.82717181, -0.31231438, 0.07307673),
  (half3)(-0.5743977, 1.36858544, -0.53183455),
  (half3)(-0.25277411, -0.05627105, 1.45875782),
};

// tone mapping params
const half cpk = 0.75;
const half cpb = 0.125;
const half cpxk = 0.0025;
const half cpxb = 0.01;

half mf(half x, half cp) {
  half rk = 9 - 100*cp;
  if (x > cp) {
    return (rk * (x-cp) * (1-(cpk*cp+cpb)) * (1+1/(rk*(1-cp))) / (1+rk*(x-cp))) + cpk*cp + cpb;
  } else if (x < cp) {
    return (rk * (x-cp) * (cpk*cp+cpb) * (1+1/(rk*cp)) / (1-rk*(x-cp))) + cpk*cp + cpb;
  } else {
    return x;
  }
}

half3 color_correct(half3 rgb, int ggain) {
  half3 ret = (0,0,0);
  half cpx = 0.01; //clamp(0.01h, 0.05h, cpxb + cpxk * min(10, ggain));
  ret += (half)rgb.x * color_correction[0];
  ret += (half)rgb.y * color_correction[1];
  ret += (half)rgb.z * color_correction[2];
  ret.x = mf(ret.x, cpx);
  ret.y = mf(ret.y, cpx);
  ret.z = mf(ret.z, cpx);
  ret = clamp(0.0h, 255.0h, ret*255.0h);
  return ret;
}

half val_from_10(const uchar * source, int gx, int gy) {
  // parse 10bit
  int start = gy * FRAME_STRIDE + (5 * (gx / 4));
  int offset = gx % 4;
  uint major = (uint)source[start + offset] << 2;
  uint minor = (source[start + 4] >> (2 * offset)) & 3;
  half pv = (half)(major + minor);

  // normalize
  pv = max(0.0h, pv - black_level);
  pv *= 0.00101833h; // /= (1024.0f - black_level);

  // correct vignetting
  if (CAM_NUM == 1) { // fcamera
    gx = (gx - RGB_WIDTH/2);
    gy = (gy - RGB_HEIGHT/2);
    float r = gx*gx + gy*gy;
    half s;
    if (r < 62500) {
      s = (half)(1.0f + 0.0000008f*r);
    } else if (r < 490000) {
      s = (half)(0.9625f + 0.0000014f*r);
    } else if (r < 1102500) {
      s = (half)(1.26434f + 0.0000000000016f*r*r);
    } else {
      s = (half)(0.53503625f + 0.0000000000022f*r*r);
    }
    pv = s * pv;
  }

  pv = clamp(0.0h, 1.0h, pv);
  return pv;
}

half fabs_diff(half x, half y) {
  return fabs(x-y);
}

half phi(half x) {
  // detection funtion
  return 2 - x;
  // if (x > 1) {
  //   return 1 / x;
  // } else {
  //   return 2 - x;
  // }
}

__kernel void debayer10(const __global uchar * in,
                        __global uchar * out,
                        __local half * cached,
                        uint ggain
                       )
{
  const int x_global = get_global_id(0);
  const int y_global = get_global_id(1);

  const int localRowLen = 2 + get_local_size(0); // 2 padding
  const int x_local = get_local_id(0); // 0-15
  const int y_local = get_local_id(1); // 0-15
  const int localOffset = (y_local + 1) * localRowLen + x_local + 1; // max 18x18-1

  int out_idx = 3 * x_global + 3 * y_global * RGB_WIDTH;

  half pv = val_from_10(in, x_global, y_global);
  cached[localOffset] = pv;

  // don't care
  if (x_global < 1 || x_global >= RGB_WIDTH - 1 || y_global < 1 || y_global >= RGB_HEIGHT - 1) {
    return;
  }

  // cache padding
  int localColOffset = -1;
  int globalColOffset = -1;

  // cache padding
  if (x_local < 1) {
    localColOffset = x_local;
    globalColOffset = -1;
    cached[(y_local + 1) * localRowLen + x_local] = val_from_10(in, x_global-1, y_global);
  } else if (x_local >= get_local_size(0) - 1) {
    localColOffset = x_local + 2;
    globalColOffset = 1;
    cached[localOffset + 1] = val_from_10(in, x_global+1, y_global);
  }

  if (y_local < 1) {
    cached[y_local * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global-1);
    if (localColOffset != -1) {
      cached[y_local * localRowLen + localColOffset] = val_from_10(in, x_global+globalColOffset, y_global-1);
    }
  } else if (y_local >= get_local_size(1) - 1) {
    cached[(y_local + 2) * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global+1);
    if (localColOffset != -1) {
      cached[(y_local + 2) * localRowLen + localColOffset] = val_from_10(in, x_global+globalColOffset, y_global+1);
    }
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  half d1 = cached[localOffset - localRowLen - 1];
  half d2 = cached[localOffset - localRowLen + 1];
  half d3 = cached[localOffset + localRowLen - 1];
  half d4 = cached[localOffset + localRowLen + 1];
  half n1 = cached[localOffset - localRowLen];
  half n2 = cached[localOffset + 1];
  half n3 = cached[localOffset + localRowLen];
  half n4 = cached[localOffset - 1];

  half3 rgb;

  // a simplified version of https://opensignalprocessingjournal.com/contents/volumes/V6/TOSIGPJ-6-1/TOSIGPJ-6-1.pdf
  if (x_global % 2 == 0) {
    if (y_global % 2 == 0) {
      rgb.y = pv; // G1(R)
      half k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      half k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      half k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      half k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G1
      rgb.x = (k2*n2+k4*n4)/(k2+k4);
      // B_G1
      rgb.z = (k1*n1+k3*n3)/(k1+k3);
    } else {
      rgb.z = pv; // B
      half k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      half k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      half k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      half k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_B
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // R_B
      rgb.x = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    }
  } else {
    if (y_global % 2 == 0) {
      rgb.x = pv; // R
      half k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      half k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      half k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      half k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_R
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // B_R
      rgb.z = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    } else {
      rgb.y = pv; // G2(B)
      half k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      half k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      half k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      half k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G2
      rgb.x = (k1*n1+k3*n3)/(k1+k3);
      // B_G2
      rgb.z = (k2*n2+k4*n4)/(k2+k4);
    }
  }

  rgb = clamp(0.0h, 1.0h, rgb);
  rgb = color_correct(rgb, (int)ggain);

  // overwrite
  int idx = x_global / (RGB_WIDTH / 6) + 6 * (y_global / (RGB_HEIGHT / 4));
  rgb = 255.0h * color_checker[idx];

  out[out_idx + 0] = (uchar)(rgb.z);
  out[out_idx + 1] = (uchar)(rgb.y);
  out[out_idx + 2] = (uchar)(rgb.x);

}
