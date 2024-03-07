#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

namespace madWarp {

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t maxEpisodeLength;
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
    };

    Manager(const Config &cfg);
    ~Manager();

    void init();
    void processActions();
    void postPhysics();

#ifdef MADRONA_CUDA_SUPPORT
    void processActionsAsync(cudaStream_t strm);
    void postPhysicsAsync(cudaStream_t strm);
#endif

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    madrona::py::Tensor resetTensor() const;
    madrona::py::Tensor doneTensor() const;

    madrona::py::Tensor actionTensor() const;
    madrona::py::Tensor rewardTensor() const;

    madrona::py::Tensor rigidBodyPositionsTensor() const;
    madrona::py::Tensor rigidBodyRotationsTensor() const;
    madrona::py::Tensor jointForcesTensor() const;

    madrona::py::Tensor rgbTensor() const;
    madrona::py::Tensor depthTensor() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    void triggerReset(int32_t world_idx);
    void setAction(int32_t world_idx, int32_t move_amount);

    madrona::render::RenderManager & getRenderManager();

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
