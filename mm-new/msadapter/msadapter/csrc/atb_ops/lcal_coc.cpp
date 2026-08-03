// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "ms_extension/api.h"

using LinearParallelParam = atb::infer::LinearParallelParam;

namespace atb {
    template <>
    struct HashOpParam<LinearParallelParam> {
        void operator()(const LinearParallelParam &param) const {
            add_param_to_buf("transWeight", param.transWeight);
            add_param_to_buf("rank", param.rank);
            add_param_to_buf("rankSize", param.rankSize);
            add_param_to_buf("rankRoot", param.rankRoot);
            add_param_to_buf("hasResidual", param.hasResidual);
            add_param_to_buf("backend", param.backend);
            add_param_to_buf("commMode", param.commMode);
            add_param_to_buf("type", param.type);
            add_param_to_buf("keepIntermediate", param.keepIntermediate);
            add_param_to_buf("commDomain", param.commDomain);
        }
    };
}  // namespace atb

void matmul_all_reduce(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                       ms::Tensor output, int rank, int rankSize, const std::string &commDomain)
{
    ms::Tensor bias = biasOpt.has_value() ? biasOpt.value() : ms::Tensor();

    LinearParallelParam param;
    bool transB = (input1.shape()[1] != input2.shape()[0]);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = rankSize;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE;
    param.keepIntermediate = false;
    param.commDomain = commDomain;

    std::vector<ms::Tensor> inputPara = {input1, input2};
    if (biasOpt.has_value()) {
        inputPara.push_back(bias);
    }
    ms::pynative::RunAtbOp("MatMulAllReduce", param, inputPara, {output});
}

auto pyboost_matmul_all_reduce(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                       ms::Tensor output, int rank, int rankSize, const std::string &commDomain) {
    return ms::pynative::PyboostRunner::Call<0>(matmul_all_reduce,input1, input2, biasOpt, output, rank, rankSize, commDomain);
}

void all_gather_matmul(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                       ms::Tensor output, int rank, int rankSize, const std::string &commDomain)
{
    ms::Tensor bias = biasOpt.has_value() ? biasOpt.value() : ms::Tensor();

    LinearParallelParam param;
    bool transB = (input1.shape()[1] != input2.shape()[0]);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = rankSize;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR;
    param.keepIntermediate = false;
    param.commDomain = commDomain;

    std::vector<ms::Tensor> inputPara = {input1, input2};
    if (biasOpt.has_value()) {
        inputPara.push_back(bias);
    }
    ms::pynative::RunAtbOp("AllGatherMatmul", param, inputPara, {output});
}

auto pyboost_all_gather_matmul(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                       ms::Tensor output, int rank, int rankSize, const std::string &commDomain) {
    return ms::pynative::PyboostRunner::Call<0>(all_gather_matmul, input1, input2, biasOpt, output, rank, rankSize, commDomain);
}

void all_gather_matmul_v2(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                          ms::Tensor output, ms::Tensor commOutput, int rank, int rankSize, const std::string &commDomain)
{
    ms::Tensor bias = biasOpt.has_value() ? biasOpt.value() : ms::Tensor();

    LinearParallelParam param;
    bool transB = (input1.shape()[1] != input2.shape()[0]);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = rankSize;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR;
    param.keepIntermediate = true;
    param.commDomain = commDomain;

    std::vector<ms::Tensor> inputPara = {input1, input2};
    if (biasOpt.has_value()) {
        inputPara.push_back(bias);
    }
    ms::pynative::RunAtbOp("AllGatherMatmulV2", param, inputPara, {output, commOutput});
}

auto pyboost_all_gather_matmul_v2(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                       ms::Tensor output, ms::Tensor commOutput, int rank, int rankSize, const std::string &commDomain) {
    return ms::pynative::PyboostRunner::Call<0>(all_gather_matmul_v2, input1, input2, biasOpt, output, commOutput, rank, rankSize, commDomain);
}


void matmul_reduce_scatter(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                           ms::Tensor output, int rank, int rankSize, const std::string &commDomain)
{
    ms::Tensor bias = biasOpt.has_value() ? biasOpt.value() : ms::Tensor();

    LinearParallelParam param;
    bool transB = (input1.shape()[1] != input2.shape()[0]);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = rankSize;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER;
    param.keepIntermediate = false;
    param.commDomain = commDomain;

    std::vector<ms::Tensor> inputPara = {input1, input2};
    if (biasOpt.has_value()) {
        inputPara.push_back(bias);
    }
    ms::pynative::RunAtbOp("MatmulReduceScatter", param, inputPara, {output});
}

auto pyboost_matmul_reduce_scatter(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                       ms::Tensor output, int rank, int rankSize, const std::string &commDomain) {
    return ms::pynative::PyboostRunner::Call<0>(matmul_reduce_scatter, input1, input2, biasOpt, output, rank, rankSize, commDomain);
}

void pure_matmul(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                 ms::Tensor output, int rank, int rankSize, const std::string &commDomain)
{
    ms::Tensor bias = biasOpt.has_value() ? biasOpt.value() : ms::Tensor();

    LinearParallelParam param;
    bool transB = (input1.shape()[1] != input2.shape()[0]);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = rankSize;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR;
    param.keepIntermediate = false;
    param.commDomain = commDomain;

    std::vector<ms::Tensor> inputPara = {input1, input2};
    if (biasOpt.has_value()) {
        inputPara.push_back(bias);
    }
    ms::pynative::RunAtbOp("PureMatmul", param, inputPara, {output});
}

auto pyboost_pure_matmul(const ms::Tensor &input1, const ms::Tensor &input2, const std::optional<ms::Tensor> &biasOpt,
                       ms::Tensor output, int rank, int rankSize, const std::string &commDomain) {
    return ms::pynative::PyboostRunner::Call<0>(pure_matmul, input1, input2, biasOpt, output, rank, rankSize, commDomain);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
    m.def("matmul_all_reduce", &pyboost_matmul_all_reduce, "matmul_all_reduce on ascend device", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"), pybind11::arg("rankSize"), pybind11::arg("commDomain"));
    m.def("all_gather_matmul", &pyboost_all_gather_matmul, "all_gather_matmul on ascend device", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"), pybind11::arg("rankSize"), pybind11::arg("commDomain"));
    m.def("all_gather_matmul_v2", &pyboost_all_gather_matmul_v2, "all_gather_matmul_v2 on ascend device", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("commOutput"),
          pybind11::arg("rank"), pybind11::arg("rankSize"), pybind11::arg("commDomain"));
    m.def("matmul_reduce_scatter", &pyboost_matmul_reduce_scatter, "matmul_reduce_scatter on ascend device", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"), pybind11::arg("rankSize"), pybind11::arg("commDomain"));
    m.def("pure_matmul", &pyboost_pure_matmul, "pure_matmul on ascend device", pybind11::arg("input1"), pybind11::arg("input2"),
          pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"), pybind11::arg("rankSize"), pybind11::arg("commDomain"));
}
