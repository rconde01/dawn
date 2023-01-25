// Copyright 2023 The Tint Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/tint/ast/test_helper.h"

namespace tint::ast {
namespace {

using namespace tint::number_suffixes;  // NOLINT
using DiagnosticAttributeTest = TestHelper;

TEST_F(DiagnosticAttributeTest, Creation) {
    auto* name = Expr("foo");
    auto* d = DiagnosticAttribute(DiagnosticSeverity::kWarning, name);
    EXPECT_EQ(d->Name(), "diagnostic");
    EXPECT_EQ(d->control->severity, DiagnosticSeverity::kWarning);
    EXPECT_EQ(d->control->rule_name, name);
}

}  // namespace
}  // namespace tint::ast