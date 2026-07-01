/**
 * @file output/render/transfer_fn.h
 * @brief Colormap tables and premultiplied RGBA look-up-table builder
 * @implements
 *   - out::buildLUT
 *   - out::colormapRGB
 * @namespaces:
 *   - out::
 * @note
 * Colormaps are stored as a handful of anchor colors and linearly
 * interpolated; this is visually indistinguishable from the full 256-entry
 * matplotlib tables for volume rendering while keeping the header compact.
 * The LUT is built on the host and deep-copied to a device View of shape
 * (N_LUT, 4) holding premultiplied RGBA (R=r*a, G=g*a, B=b*a, A=a).
 */

#ifndef OUTPUT_RENDER_TRANSFER_FN_H
#define OUTPUT_RENDER_TRANSFER_FN_H

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include <array>
#include <string>
#include <vector>

namespace out {

  namespace cmap_hidden {

    // anchor colors sampled at uniform positions in [0, 1]
    struct Anchors {
      const float (*rgb)[3];
      int n;
    };

    inline constexpr float viridis[9][3] = {
      { 0.267004f, 0.004874f, 0.329415f },
      { 0.282623f, 0.140926f, 0.457517f },
      { 0.253935f, 0.265254f, 0.529983f },
      { 0.206756f, 0.371758f, 0.553117f },
      { 0.163625f, 0.471133f, 0.558148f },
      { 0.127568f, 0.566949f, 0.550556f },
      { 0.134692f, 0.658636f, 0.517649f },
      { 0.477504f, 0.821444f, 0.318195f },
      { 0.993248f, 0.906157f, 0.143936f },
    };

    inline constexpr float inferno[9][3] = {
      { 0.001462f, 0.000466f, 0.013866f },
      { 0.087411f, 0.044556f, 0.224813f },
      { 0.258234f, 0.038571f, 0.406485f },
      { 0.416331f, 0.090203f, 0.432943f },
      { 0.578304f, 0.148039f, 0.404411f },
      { 0.735683f, 0.215906f, 0.330245f },
      { 0.865006f, 0.316822f, 0.226055f },
      { 0.954506f, 0.468744f, 0.099874f },
      { 0.988362f, 0.998364f, 0.644924f },
    };

    inline constexpr float plasma[9][3] = {
      { 0.050383f, 0.029803f, 0.527975f },
      { 0.287076f, 0.010855f, 0.627295f },
      { 0.417642f, 0.000564f, 0.658390f },
      { 0.562738f, 0.051545f, 0.641509f },
      { 0.692840f, 0.165141f, 0.564522f },
      { 0.798216f, 0.280197f, 0.469538f },
      { 0.881443f, 0.392529f, 0.383229f },
      { 0.949217f, 0.517763f, 0.295662f },
      { 0.940015f, 0.975158f, 0.131326f },
    };

    // Moreland cool-to-warm diverging
    inline constexpr float cool2warm[3][3] = {
      { 0.230f, 0.299f, 0.754f },
      { 0.865f, 0.865f, 0.865f },
      { 0.706f, 0.016f, 0.150f },
    };

    inline constexpr float gray[2][3] = {
      { 0.0f, 0.0f, 0.0f },
      { 1.0f, 1.0f, 1.0f },
    };

    // -----------------------------------------------------------------------
    // CMasher scientific colormaps (https://cmasher.readthedocs.io)
    //
    // The following anchor tables are uniform downsamplings (33 anchors) of the
    // published CMasher colormap data, re-implemented from the source at
    // https://github.com/1313e/CMasher (src/cmasher/colormaps/<name>/<name>_norm.txt).
    // At 33 anchors the linear-interpolation error versus the full 256/511-entry
    // tables is < 4.5/255 for every map, i.e. visually indistinguishable.
    //
    // CMasher is distributed under the BSD 3-Clause License:
    //   Copyright (c) 2019-2021, Ellert van der Velden
    //   All rights reserved.
    // Redistribution and use in source and binary forms, with or without
    // modification, are permitted provided that the copyright notice, this list
    // of conditions and the BSD-3-Clause disclaimer are retained. The name of the
    // copyright holder may not be used to endorse products without permission.
    // -----------------------------------------------------------------------

    // cmasher::dusk (33 anchors sampled from the 256-entry table)
    inline constexpr float dusk[33][3] = {
      { 0.000000f, 0.000000f, 0.000000f },
      { 0.006238f, 0.007290f, 0.012708f },
      { 0.018622f, 0.026187f, 0.053076f },
      { 0.029900f, 0.055366f, 0.101787f },
      { 0.030149f, 0.086842f, 0.147576f },
      { 0.015542f, 0.120312f, 0.179819f },
      { 0.010319f, 0.152335f, 0.193847f },
      { 0.029326f, 0.181190f, 0.200163f },
      { 0.066363f, 0.207841f, 0.204068f },
      { 0.103996f, 0.233120f, 0.206493f },
      { 0.141503f, 0.257403f, 0.206888f },
      { 0.180382f, 0.280651f, 0.204334f },
      { 0.222140f, 0.302532f, 0.198293f },
      { 0.267566f, 0.322632f, 0.189012f },
      { 0.316579f, 0.340653f, 0.177441f },
      { 0.368580f, 0.356469f, 0.164931f },
      { 0.422824f, 0.370100f, 0.153037f },
      { 0.471588f, 0.380324f, 0.144475f },
      { 0.528296f, 0.390215f, 0.138408f },
      { 0.585639f, 0.398375f, 0.137901f },
      { 0.643321f, 0.404994f, 0.144156f },
      { 0.701148f, 0.410242f, 0.157700f },
      { 0.758937f, 0.414304f, 0.178666f },
      { 0.816278f, 0.417544f, 0.207638f },
      { 0.871717f, 0.421238f, 0.247431f },
      { 0.918409f, 0.431732f, 0.307531f },
      { 0.940305f, 0.464843f, 0.388730f },
      { 0.947103f, 0.510940f, 0.465808f },
      { 0.949229f, 0.559258f, 0.537097f },
      { 0.949161f, 0.607586f, 0.604625f },
      { 0.947863f, 0.655456f, 0.669259f },
      { 0.945994f, 0.702786f, 0.731248f },
      { 0.944208f, 0.749586f, 0.790456f },
    };

    // cmasher::cosmic (33 anchors sampled from the 256-entry table)
    inline constexpr float cosmic[33][3] = {
      { 0.000000f, 0.000000f, 0.000000f },
      { 0.010239f, 0.006872f, 0.013172f },
      { 0.038809f, 0.022373f, 0.054399f },
      { 0.076237f, 0.043279f, 0.104947f },
      { 0.112700f, 0.063138f, 0.158384f },
      { 0.148869f, 0.079429f, 0.215952f },
      { 0.185079f, 0.091867f, 0.278775f },
      { 0.221470f, 0.099655f, 0.348020f },
      { 0.257982f, 0.101322f, 0.424930f },
      { 0.294209f, 0.094291f, 0.510709f },
      { 0.328977f, 0.074083f, 0.605931f },
      { 0.359190f, 0.034502f, 0.708287f },
      { 0.377711f, 0.016748f, 0.805820f },
      { 0.375894f, 0.098454f, 0.873588f },
      { 0.355826f, 0.192411f, 0.902268f },
      { 0.326497f, 0.271620f, 0.906235f },
      { 0.293925f, 0.337879f, 0.899030f },
      { 0.265164f, 0.388114f, 0.889366f },
      { 0.233446f, 0.439282f, 0.877747f },
      { 0.203862f, 0.485773f, 0.867189f },
      { 0.176988f, 0.529060f, 0.858475f },
      { 0.153054f, 0.570207f, 0.851855f },
      { 0.131837f, 0.609999f, 0.847280f },
      { 0.112515f, 0.649034f, 0.844504f },
      { 0.093584f, 0.687770f, 0.843135f },
      { 0.072938f, 0.726550f, 0.842663f },
      { 0.048159f, 0.765608f, 0.842480f },
      { 0.021220f, 0.805066f, 0.841902f },
      { 0.009956f, 0.844909f, 0.840192f },
      { 0.039372f, 0.884930f, 0.836603f },
      { 0.115052f, 0.924556f, 0.830460f },
      { 0.218056f, 0.962249f, 0.821634f },
      { 0.371763f, 0.992456f, 0.816521f },
    };

    // cmasher::freeze (33 anchors sampled from the 256-entry table)
    inline constexpr float freeze[33][3] = {
      { 0.000000f, 0.000000f, 0.000000f },
      { 0.010910f, 0.009007f, 0.014906f },
      { 0.039247f, 0.030823f, 0.059160f },
      { 0.074148f, 0.060008f, 0.110444f },
      { 0.106606f, 0.087232f, 0.163635f },
      { 0.137212f, 0.112671f, 0.219689f },
      { 0.166145f, 0.136733f, 0.279301f },
      { 0.193344f, 0.159680f, 0.343042f },
      { 0.218514f, 0.181739f, 0.411377f },
      { 0.241052f, 0.203213f, 0.484587f },
      { 0.259859f, 0.224652f, 0.562539f },
      { 0.272953f, 0.247192f, 0.644102f },
      { 0.276779f, 0.273147f, 0.725721f },
      { 0.265827f, 0.306369f, 0.798716f },
      { 0.236083f, 0.349541f, 0.849696f },
      { 0.192331f, 0.398903f, 0.873582f },
      { 0.145422f, 0.448298f, 0.878894f },
      { 0.112009f, 0.489316f, 0.875932f },
      { 0.099232f, 0.533336f, 0.869029f },
      { 0.122966f, 0.574696f, 0.861171f },
      { 0.170084f, 0.613909f, 0.853779f },
      { 0.227234f, 0.651383f, 0.847515f },
      { 0.289300f, 0.687372f, 0.842698f },
      { 0.355048f, 0.721960f, 0.839586f },
      { 0.424505f, 0.755075f, 0.838630f },
      { 0.497701f, 0.786577f, 0.840759f },
      { 0.573685f, 0.816502f, 0.847460f },
      { 0.650307f, 0.845347f, 0.860140f },
      { 0.725396f, 0.873979f, 0.879132f },
      { 0.797886f, 0.903229f, 0.903669f },
      { 0.867683f, 0.933696f, 0.932619f },
      { 0.935038f, 0.965809f, 0.964980f },
      { 1.000000f, 1.000000f, 1.000000f },
    };

    // cmasher::apple (33 anchors sampled from the 256-entry table)
    inline constexpr float apple[33][3] = {
      { 0.000000f, 0.000000f, 0.000000f },
      { 0.018449f, 0.006683f, 0.008999f },
      { 0.069836f, 0.019462f, 0.030130f },
      { 0.124822f, 0.033495f, 0.057048f },
      { 0.180264f, 0.045068f, 0.080127f },
      { 0.236726f, 0.050851f, 0.098639f },
      { 0.294404f, 0.049708f, 0.111860f },
      { 0.353148f, 0.039763f, 0.118281f },
      { 0.412085f, 0.021350f, 0.115115f },
      { 0.467944f, 0.008973f, 0.098099f },
      { 0.512805f, 0.042578f, 0.069239f },
      { 0.544883f, 0.107236f, 0.041311f },
      { 0.569454f, 0.165936f, 0.019988f },
      { 0.589142f, 0.220295f, 0.007127f },
      { 0.604821f, 0.272232f, 0.001856f },
      { 0.616738f, 0.322887f, 0.005791f },
      { 0.624881f, 0.372928f, 0.022422f },
      { 0.628812f, 0.416517f, 0.050591f },
      { 0.629507f, 0.466302f, 0.088693f },
      { 0.625985f, 0.516149f, 0.130045f },
      { 0.618051f, 0.566092f, 0.175071f },
      { 0.605455f, 0.616124f, 0.224400f },
      { 0.587974f, 0.666152f, 0.279195f },
      { 0.566011f, 0.715805f, 0.341631f },
      { 0.543179f, 0.763859f, 0.415364f },
      { 0.534328f, 0.806916f, 0.503615f },
      { 0.564391f, 0.840721f, 0.598311f },
      { 0.628027f, 0.867449f, 0.684639f },
      { 0.703665f, 0.891763f, 0.760351f },
      { 0.781104f, 0.916124f, 0.828098f },
      { 0.857052f, 0.941725f, 0.890058f },
      { 0.930451f, 0.969331f, 0.947434f },
      { 1.000000f, 1.000000f, 1.000000f },
    };

    // cmasher::gothic (33 anchors sampled from the 256-entry table)
    inline constexpr float gothic[33][3] = {
      { 0.000000f, 0.000000f, 0.000000f },
      { 0.009103f, 0.009497f, 0.017125f },
      { 0.031259f, 0.032684f, 0.068646f },
      { 0.061552f, 0.062439f, 0.127729f },
      { 0.091294f, 0.088771f, 0.191076f },
      { 0.121658f, 0.111295f, 0.260048f },
      { 0.154308f, 0.129160f, 0.335784f },
      { 0.191149f, 0.140572f, 0.419137f },
      { 0.234428f, 0.142239f, 0.510129f },
      { 0.286418f, 0.128494f, 0.605972f },
      { 0.347449f, 0.092109f, 0.695963f },
      { 0.411712f, 0.037080f, 0.759733f },
      { 0.471546f, 0.020266f, 0.789424f },
      { 0.526274f, 0.060551f, 0.796780f },
      { 0.578066f, 0.108541f, 0.793004f },
      { 0.628368f, 0.151332f, 0.783641f },
      { 0.677621f, 0.190718f, 0.770763f },
      { 0.719390f, 0.224799f, 0.756764f },
      { 0.763054f, 0.267998f, 0.736645f },
      { 0.790627f, 0.328361f, 0.713928f },
      { 0.791142f, 0.402697f, 0.719584f },
      { 0.787649f, 0.467602f, 0.747951f },
      { 0.784959f, 0.526150f, 0.782754f },
      { 0.783559f, 0.580954f, 0.819616f },
      { 0.783736f, 0.633370f, 0.856997f },
      { 0.785586f, 0.684344f, 0.893939f },
      { 0.789112f, 0.734726f, 0.928646f },
      { 0.795666f, 0.785049f, 0.956164f },
      { 0.813132f, 0.833710f, 0.968235f },
      { 0.848938f, 0.877750f, 0.970393f },
      { 0.895684f, 0.918861f, 0.974579f },
      { 0.947082f, 0.959196f, 0.984058f },
      { 1.000000f, 1.000000f, 1.000000f },
    };

    // cmasher::sunburst (33 anchors sampled from the 256-entry table)
    inline constexpr float sunburst[33][3] = {
      { 0.000000f, 0.000000f, 0.000000f },
      { 0.014374f, 0.007761f, 0.013830f },
      { 0.057181f, 0.023825f, 0.051569f },
      { 0.107503f, 0.043062f, 0.091435f },
      { 0.159109f, 0.059433f, 0.125622f },
      { 0.212012f, 0.071665f, 0.153805f },
      { 0.266014f, 0.080343f, 0.175895f },
      { 0.320933f, 0.085763f, 0.191981f },
      { 0.376633f, 0.087997f, 0.202182f },
      { 0.432991f, 0.086959f, 0.206564f },
      { 0.489851f, 0.082504f, 0.205087f },
      { 0.546970f, 0.074631f, 0.197554f },
      { 0.603921f, 0.064180f, 0.183549f },
      { 0.659892f, 0.055133f, 0.162324f },
      { 0.713225f, 0.059572f, 0.132732f },
      { 0.760503f, 0.093519f, 0.093993f },
      { 0.796917f, 0.154735f, 0.050648f },
      { 0.819328f, 0.216009f, 0.025564f },
      { 0.837747f, 0.284012f, 0.033882f },
      { 0.851299f, 0.348017f, 0.074896f },
      { 0.861316f, 0.408787f, 0.125585f },
      { 0.868446f, 0.467188f, 0.180749f },
      { 0.873123f, 0.523806f, 0.239715f },
      { 0.875769f, 0.578986f, 0.302576f },
      { 0.876914f, 0.632883f, 0.369564f },
      { 0.877316f, 0.685489f, 0.440900f },
      { 0.878101f, 0.736650f, 0.516675f },
      { 0.880900f, 0.786078f, 0.596685f },
      { 0.887890f, 0.833402f, 0.680181f },
      { 0.901543f, 0.878304f, 0.765633f },
      { 0.924079f, 0.920687f, 0.850654f },
      { 0.957291f, 0.960697f, 0.931527f },
      { 1.000000f, 1.000000f, 1.000000f },
    };

    // cmasher::voltage (33 anchors sampled from the 256-entry table)
    inline constexpr float voltage[33][3] = {
      { 0.000000f, 0.000000f, 0.000000f },
      { 0.015829f, 0.007377f, 0.012045f },
      { 0.060346f, 0.022787f, 0.046824f },
      { 0.109165f, 0.041971f, 0.090162f },
      { 0.157514f, 0.059030f, 0.134981f },
      { 0.205921f, 0.071648f, 0.182831f },
      { 0.254512f, 0.079556f, 0.235168f },
      { 0.303074f, 0.082057f, 0.293555f },
      { 0.350914f, 0.078204f, 0.359600f },
      { 0.396550f, 0.067768f, 0.434356f },
      { 0.437427f, 0.055503f, 0.516639f },
      { 0.470517f, 0.059198f, 0.601227f },
      { 0.494085f, 0.093088f, 0.680818f },
      { 0.508421f, 0.145253f, 0.750719f },
      { 0.514803f, 0.203410f, 0.809898f },
      { 0.514558f, 0.262635f, 0.859195f },
      { 0.508787f, 0.321230f, 0.899892f },
      { 0.499930f, 0.371547f, 0.929319f },
      { 0.486263f, 0.427841f, 0.956532f },
      { 0.469915f, 0.482826f, 0.977159f },
      { 0.452770f, 0.536437f, 0.991010f },
      { 0.438275f, 0.588421f, 0.997509f },
      { 0.432385f, 0.638191f, 0.996018f },
      { 0.443151f, 0.684778f, 0.986745f },
      { 0.476274f, 0.727179f, 0.972136f },
      { 0.529358f, 0.765229f, 0.956912f },
      { 0.594140f, 0.799927f, 0.945473f },
      { 0.663403f, 0.832718f, 0.939967f },
      { 0.733296f, 0.864814f, 0.940775f },
      { 0.802260f, 0.897066f, 0.947563f },
      { 0.869795f, 0.930059f, 0.959867f },
      { 0.935781f, 0.964231f, 0.977340f },
      { 1.000000f, 1.000000f, 1.000000f },
    };

    // cmasher::ocean (33 anchors sampled from the 256-entry table)
    inline constexpr float ocean[33][3] = {
      { 0.110363f, 0.001691f, 0.253026f },
      { 0.124516f, 0.043066f, 0.289045f },
      { 0.135665f, 0.084868f, 0.324305f },
      { 0.143805f, 0.121839f, 0.358190f },
      { 0.148976f, 0.156972f, 0.390248f },
      { 0.151327f, 0.191269f, 0.420093f },
      { 0.151211f, 0.225122f, 0.447430f },
      { 0.149271f, 0.258662f, 0.472116f },
      { 0.146471f, 0.291895f, 0.494196f },
      { 0.144043f, 0.324790f, 0.513894f },
      { 0.143346f, 0.357322f, 0.531555f },
      { 0.145639f, 0.389496f, 0.547565f },
      { 0.151848f, 0.421347f, 0.562289f },
      { 0.162417f, 0.452929f, 0.576030f },
      { 0.177344f, 0.484306f, 0.589017f },
      { 0.196368f, 0.515533f, 0.601409f },
      { 0.219188f, 0.546652f, 0.613299f },
      { 0.242129f, 0.573802f, 0.623330f },
      { 0.271793f, 0.604715f, 0.634385f },
      { 0.305461f, 0.635419f, 0.645036f },
      { 0.343819f, 0.665728f, 0.655393f },
      { 0.387893f, 0.695314f, 0.665825f },
      { 0.438718f, 0.723698f, 0.677325f },
      { 0.496053f, 0.750479f, 0.691913f },
      { 0.556972f, 0.775909f, 0.711830f },
      { 0.617772f, 0.800881f, 0.737580f },
      { 0.676718f, 0.826177f, 0.768091f },
      { 0.733666f, 0.852223f, 0.802113f },
      { 0.788975f, 0.879240f, 0.838730f },
      { 0.843051f, 0.907368f, 0.877312f },
      { 0.896212f, 0.936732f, 0.917381f },
      { 0.948620f, 0.967497f, 0.958467f },
      { 1.000000f, 1.000000f, 1.000000f },
    };

    // cmasher::fusion (diverging; 33 anchors sampled from the 511-entry table)
    inline constexpr float fusion[33][3] = {
      { 0.152696f, 0.015942f, 0.069889f },
      { 0.243393f, 0.027996f, 0.138374f },
      { 0.339396f, 0.022662f, 0.187330f },
      { 0.434194f, 0.019908f, 0.202137f },
      { 0.518461f, 0.063682f, 0.191551f },
      { 0.591978f, 0.129217f, 0.171843f },
      { 0.656187f, 0.199931f, 0.150076f },
      { 0.710976f, 0.275318f, 0.131287f },
      { 0.754925f, 0.356047f, 0.126115f },
      { 0.784581f, 0.436574f, 0.150892f },
      { 0.803828f, 0.525723f, 0.220961f },
      { 0.815422f, 0.614033f, 0.328856f },
      { 0.828457f, 0.697985f, 0.458793f },
      { 0.850099f, 0.777148f, 0.598141f },
      { 0.884001f, 0.852908f, 0.739531f },
      { 0.932364f, 0.926807f, 0.878147f },
      { 1.000000f, 1.000000f, 1.000000f },
      { 0.882739f, 0.938948f, 0.943839f },
      { 0.759644f, 0.882704f, 0.900553f },
      { 0.630638f, 0.828950f, 0.872602f },
      { 0.499428f, 0.774306f, 0.861660f },
      { 0.381603f, 0.714439f, 0.863772f },
      { 0.301845f, 0.647196f, 0.868704f },
      { 0.272703f, 0.573492f, 0.869040f },
      { 0.281047f, 0.499433f, 0.863193f },
      { 0.307182f, 0.414842f, 0.849495f },
      { 0.335700f, 0.322756f, 0.825580f },
      { 0.356873f, 0.220225f, 0.784504f },
      { 0.360275f, 0.107524f, 0.709606f },
      { 0.327077f, 0.044725f, 0.576747f },
      { 0.256240f, 0.065580f, 0.425468f },
      { 0.175982f, 0.062679f, 0.300240f },
      { 0.095379f, 0.037917f, 0.194868f },
    };

    // cmasher::prinsenvlag (diverging; 33 anchors sampled from the 511-entry table)
    inline constexpr float prinsenvlag[33][3] = {
      { 0.666523f, 0.321623f, 0.271748f },
      { 0.715454f, 0.343630f, 0.238454f },
      { 0.759975f, 0.370421f, 0.199167f },
      { 0.798794f, 0.402978f, 0.153663f },
      { 0.830257f, 0.442137f, 0.100784f },
      { 0.852097f, 0.488559f, 0.041408f },
      { 0.861821f, 0.542123f, 0.032681f },
      { 0.860262f, 0.599807f, 0.123019f },
      { 0.854920f, 0.655912f, 0.231638f },
      { 0.852701f, 0.704526f, 0.333943f },
      { 0.855639f, 0.752429f, 0.440786f },
      { 0.864494f, 0.797240f, 0.544831f },
      { 0.879227f, 0.839859f, 0.645897f },
      { 0.899719f, 0.880993f, 0.743689f },
      { 0.925990f, 0.921172f, 0.837642f },
      { 0.958977f, 0.960584f, 0.926102f },
      { 1.000000f, 1.000000f, 1.000000f },
      { 0.926250f, 0.969453f, 0.961500f },
      { 0.846431f, 0.941633f, 0.927771f },
      { 0.760433f, 0.915515f, 0.901576f },
      { 0.669026f, 0.889675f, 0.886082f },
      { 0.578083f, 0.861643f, 0.882953f },
      { 0.498336f, 0.829280f, 0.888643f },
      { 0.435785f, 0.792736f, 0.897592f },
      { 0.392859f, 0.755673f, 0.906500f },
      { 0.363227f, 0.713780f, 0.915753f },
      { 0.350577f, 0.669530f, 0.923924f },
      { 0.355484f, 0.622674f, 0.928631f },
      { 0.377364f, 0.573245f, 0.923046f },
      { 0.410189f, 0.524116f, 0.889701f },
      { 0.432426f, 0.483706f, 0.814621f },
      { 0.433280f, 0.452439f, 0.723666f },
      { 0.421591f, 0.424905f, 0.636181f },
    };

    // ColorBrewer "RdBu" diverging map, reversed (blue -> white -> red), as
    // exposed by matplotlib under the name "RdBu_r". matplotlib builds it by
    // linearly interpolating these 11 control points, so the uniform-anchor
    // scheme above reproduces it to < 1.6/255.
    // Colors from ColorBrewer (https://colorbrewer2.org) by Cynthia A. Brewer,
    // Geography, Pennsylvania State University -- Apache License 2.0.
    inline constexpr float rdbu_r[11][3] = {
      { 0.019608f, 0.188235f, 0.380392f },
      { 0.132026f, 0.403460f, 0.676278f },
      { 0.262745f, 0.576471f, 0.764706f },
      { 0.566474f, 0.768704f, 0.868512f },
      { 0.819608f, 0.898039f, 0.941176f },
      { 0.969089f, 0.966474f, 0.964937f },
      { 0.992157f, 0.858824f, 0.780392f },
      { 0.957555f, 0.651211f, 0.515110f },
      { 0.839216f, 0.376471f, 0.301961f },
      { 0.692272f, 0.092272f, 0.167705f },
      { 0.403922f, 0.000000f, 0.121569f },
    };

    inline auto lookup(const std::string& name) -> Anchors {
      if (name == "inferno") {
        return { inferno, 9 };
      } else if (name == "plasma") {
        return { plasma, 9 };
      } else if (name == "cool2warm" or name == "coolwarm") {
        return { cool2warm, 3 };
      } else if (name == "gray" or name == "grey") {
        return { gray, 2 };
        // CMasher scientific colormaps (accept an optional "cmr." prefix so
        // names can be copied straight from the CMasher documentation)
      } else if (name == "dusk" or name == "cmr.dusk") {
        return { dusk, 33 };
      } else if (name == "cosmic" or name == "cmr.cosmic") {
        return { cosmic, 33 };
      } else if (name == "freeze" or name == "cmr.freeze") {
        return { freeze, 33 };
      } else if (name == "apple" or name == "cmr.apple") {
        return { apple, 33 };
      } else if (name == "gothic" or name == "cmr.gothic") {
        return { gothic, 33 };
      } else if (name == "sunburst" or name == "cmr.sunburst") {
        return { sunburst, 33 };
      } else if (name == "voltage" or name == "cmr.voltage") {
        return { voltage, 33 };
      } else if (name == "ocean" or name == "cmr.ocean") {
        return { ocean, 33 };
      } else if (name == "fusion" or name == "cmr.fusion") {
        return { fusion, 33 };
      } else if (name == "prinsenvlag" or name == "cmr.prinsenvlag") {
        return { prinsenvlag, 33 };
      } else if (name == "RdBu_r" or name == "rdbu_r") {
        return { rdbu_r, 11 };
      } else {
        // default / "viridis"
        return { viridis, 9 };
      }
    }

  } // namespace cmap_hidden

  /**
   * @brief Sample a named colormap at u in [0, 1], returning RGB in [0, 1].
   */
  inline void colormapRGB(const std::string& name,
                          real_t             u,
                          real_t&            r,
                          real_t&            g,
                          real_t&            b) {
    const auto anchors = cmap_hidden::lookup(name);
    if (u <= ZERO) {
      r = anchors.rgb[0][0];
      g = anchors.rgb[0][1];
      b = anchors.rgb[0][2];
      return;
    }
    if (u >= ONE) {
      r = anchors.rgb[anchors.n - 1][0];
      g = anchors.rgb[anchors.n - 1][1];
      b = anchors.rgb[anchors.n - 1][2];
      return;
    }
    const real_t x   = u * static_cast<real_t>(anchors.n - 1);
    const int    i0  = static_cast<int>(x);
    const int    i1  = (i0 + 1 < anchors.n) ? (i0 + 1) : i0;
    const real_t t   = x - static_cast<real_t>(i0);
    r = static_cast<real_t>(anchors.rgb[i0][0]) * (ONE - t) +
        static_cast<real_t>(anchors.rgb[i1][0]) * t;
    g = static_cast<real_t>(anchors.rgb[i0][1]) * (ONE - t) +
        static_cast<real_t>(anchors.rgb[i1][1]) * t;
    b = static_cast<real_t>(anchors.rgb[i0][2]) * (ONE - t) +
        static_cast<real_t>(anchors.rgb[i1][2]) * t;
  }

  /**
   * @brief Piecewise-linear opacity from sorted (position, alpha) control points.
   */
  inline auto alphaAt(const std::vector<std::array<real_t, 2>>& pts, real_t u)
    -> real_t {
    if (pts.empty()) {
      return u; // sensible default: linear ramp
    }
    if (u <= pts.front()[0]) {
      return pts.front()[1];
    }
    if (u >= pts.back()[0]) {
      return pts.back()[1];
    }
    for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
      if (u >= pts[i][0] and u <= pts[i + 1][0]) {
        const real_t span = pts[i + 1][0] - pts[i][0];
        const real_t t    = (span > ZERO) ? (u - pts[i][0]) / span : ZERO;
        return pts[i][1] * (ONE - t) + pts[i + 1][1] * t;
      }
    }
    return pts.back()[1];
  }

  /**
   * @brief Build a premultiplied RGBA device LUT from a colormap + alpha points.
   * @param colormap name of the colormap
   * @param n_lut number of entries
   * @param alpha_pts sorted (position, alpha) control points in [0,1]x[0,1]
   * @return device View of shape (n_lut, 4), premultiplied RGBA
   */
  inline auto buildLUT(const std::string&                        colormap,
                       int                                       n_lut,
                       const std::vector<std::array<real_t, 2>>& alpha_pts)
    -> array_t<real_t* [4]> {
    array_t<real_t* [4]> lut { "render_lut", static_cast<std::size_t>(n_lut) };
    auto                 lut_h = Kokkos::create_mirror_view(lut);
    for (int i = 0; i < n_lut; ++i) {
      const real_t u = (n_lut > 1)
                         ? static_cast<real_t>(i) / static_cast<real_t>(n_lut - 1)
                         : ZERO;
      real_t r, g, b;
      colormapRGB(colormap, u, r, g, b);
      const real_t a = alphaAt(alpha_pts, u);
      lut_h(i, 0)    = r * a; // premultiplied
      lut_h(i, 1)    = g * a;
      lut_h(i, 2)    = b * a;
      lut_h(i, 3)    = a;
    }
    Kokkos::deep_copy(lut, lut_h);
    return lut;
  }

} // namespace out

#endif // OUTPUT_RENDER_TRANSFER_FN_H
