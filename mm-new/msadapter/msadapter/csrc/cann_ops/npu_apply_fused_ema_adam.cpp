#include <set>
#include <functional>
#include <optional>
#include "ms_extension/all.h"

namespace ms::pynative {
using AclnnLaunchFunc = std::function<void(mindspore::device::DeviceContext *, size_t)>;

class InnerAclnnOpRunner : public PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;
  void SetLaunchFunc(AclnnLaunchFunc func) { launch_func_ = func; }

 protected:
  using PyboostRunner::CalcWorkspace;
  void LaunchKernel() override {
    if (launch_func_ != nullptr) {
      launch_func_(_device_context_, _stream_id_);
    }
  }
  void _DispatchLaunchTask() override { LaunchKernel(); }
  AclnnLaunchFunc launch_func_{nullptr};
};

#define LAUNCH_ACLNN_FUNC(aclnn_api, ...)                                \
  [__VA_ARGS__](auto __device_context, auto __stream_id) {               \
    LAUNCH_ACLNN(aclnn_api, __device_context, __stream_id, __VA_ARGS__); \
  }
}  // namespace ms::pynative

namespace mindspore {
std::vector<ms::Tensor> npu_apply_fused_ema_adamw(ms::Tensor grad,
    ms::Tensor var,
    ms::Tensor m,
    ms::Tensor v,
    ms::Tensor s,
    ms::Tensor step,
    std::optional<double> lr,
    std::optional<double> ema_decay,
    std::optional<double> beta1,
    std::optional<double> beta2,
    std::optional<double> eps,
    std::optional<int64_t> mode,
    std::optional<bool> bias_correction,
    std::optional<double> weight_decay) {

    auto grad_t = grad.tensor();
    auto var_t = var.tensor();
    auto m_t = m.tensor();
    auto v_t = v.tensor();
    auto s_t = s.tensor();
    auto step_t = step.tensor();
    double lr_ = double(lr.value());
    double ema_decay_ = double(ema_decay.value());
    double beta1_ = double(beta1.value());
    double beta2_ = double(beta2.value());
    double eps_ = double(eps.value());
    int64_t mode_ = int64_t(mode.value());
    bool bias_correction_ = bool(bias_correction.value());
    double weight_decay_ = double(weight_decay.value());
    auto runner = std::make_shared<ms::pynative::InnerAclnnOpRunner>("ApplyFusedEmaAdam");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyFusedEmaAdam, grad_t, var_t, m_t, v_t, s_t, step_t, lr_, ema_decay_, beta1_, beta2_, eps_, mode_, bias_correction_, weight_decay_));
    runner->Run({grad, var, m, v, s, step}, {var, m, v, s});
    return runner->outputs();
}
}  // namespace mindspore

auto pyboost_npu_apply_fused_ema_adamw(ms::Tensor grad, \
    ms::Tensor var, \
    ms::Tensor m, \
    ms::Tensor v, \
    ms::Tensor s, \
    ms::Tensor step, \
    std::optional<double> lr, \
    std::optional<double> ema_decay, \
    std::optional<double> beta1, \
    std::optional<double> beta2, \
    std::optional<double> eps, \
    std::optional<int64_t> mode, \
    std::optional<bool> bias_correction, \
    std::optional<double> weight_decay) {
  return ms::pynative::PyboostRunner::Call<4>(mindspore::npu_apply_fused_ema_adamw, grad, var,m,v,s,step,lr,ema_decay,beta1,beta2,eps,mode,bias_correction,weight_decay);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
    m.def("npu_apply_fused_ema_adamw",
         &pyboost_npu_apply_fused_ema_adamw,
         "aclnnAapplyFusedEmaAdamw",
         pybind11::arg("grad"),
         pybind11::arg("var"),
         pybind11::arg("m"),
         pybind11::arg("v"),
         pybind11::arg("s"),
         pybind11::arg("step"),
         pybind11::arg("lr") = 1e-3f,
         pybind11::arg("ema_decay") = 0.9999,
         pybind11::arg("beta1") = 0.9,
         pybind11::arg("beta2") = 0.999,
         pybind11::arg("eps") = 1e-8f,
         pybind11::arg("mode") = 1,
         pybind11::arg("bias_correction") = true,
         pybind11::arg("weight_decay") = 0.0);
}
