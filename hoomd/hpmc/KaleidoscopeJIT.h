//===----- KaleidoscopeJIT.h - A simple JIT for Kaleidoscope ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains a simple JIT definition for use in the kaleidoscope tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
#define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include <memory>

#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"

namespace llvm
    {
namespace orc
    {
class KaleidoscopeJIT
    {
    public:
        ExecutionSession ES;
        RTDyldObjectLinkingLayer ObjectLayer;
        IRCompileLayer CompileLayer;

        DataLayout DL;
        MangleAndInterner Mangle;
        ThreadSafeContext Ctx;
        JITDylib *mainJD;

    KaleidoscopeJIT(JITTargetMachineBuilder JTMB, DataLayout DL)
        : ObjectLayer(ES,
                        []() { return std::make_unique<SectionMemoryManager>(); }),
            CompileLayer(ES, ObjectLayer, std::make_unique<ConcurrentIRCompiler>(ConcurrentIRCompiler(std::move(JTMB)))),
            DL(std::move(DL)), Mangle(ES, this->DL),
            Ctx(std::make_unique<LLVMContext>())
        {
        mainJD = ES.getJITDylibByName("<main>");
        if (!mainJD)
            {
            mainJD = &(ES.createJITDylib("<main>").get());
            }

        mainJD->addGenerator(
            cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL.getGlobalPrefix())));
    }
    const DataLayout &getDataLayout() const { return DL; }

  Error addModule(ThreadSafeModule TSM, ResourceTrackerSP RT = nullptr) {
    if (!RT)
      RT = mainJD->getDefaultResourceTracker();
    return CompileLayer.add(RT, std::move(TSM));
  }

  Expected<JITEvaluatedSymbol> findSymbol(std::string Name) {
    return ES.lookup({mainJD}, Mangle(Name));
  }

    JITTargetAddress getSymbolAddress(const std::string Name)
        {
        return findSymbol(Name)->getAddress();
        }
    };

    } // End namespace orc.
    } // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
