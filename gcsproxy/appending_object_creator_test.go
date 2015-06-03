// Copyright 2015 Google Inc. All Rights Reserved.
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

package gcsproxy

import (
	"errors"
	"io/ioutil"
	"strings"
	"testing"

	"github.com/jacobsa/gcloud/gcs"
	"github.com/jacobsa/gcloud/gcs/mock_gcs"
	. "github.com/jacobsa/oglematchers"
	. "github.com/jacobsa/oglemock"
	. "github.com/jacobsa/ogletest"
	"golang.org/x/net/context"
)

func TestAppendObjectCreator(t *testing.T) { RunTests(t) }

////////////////////////////////////////////////////////////////////////
// Boilerplate
////////////////////////////////////////////////////////////////////////

const prefix = ".gcsfuse_tmp/"

type AppendObjectCreatorTest struct {
	ctx     context.Context
	bucket  mock_gcs.MockBucket
	creator objectCreator

	srcObject   gcs.Object
	srcContents string
}

var _ SetUpInterface = &AppendObjectCreatorTest{}

func init() { RegisterTestSuite(&AppendObjectCreatorTest{}) }

func (t *AppendObjectCreatorTest) SetUp(ti *TestInfo) {
	t.ctx = ti.Ctx

	// Create the bucket.
	t.bucket = mock_gcs.NewMockBucket(ti.MockController, "bucket")

	// Create the creator.
	t.creator = newAppendObjectCreator(prefix, t.bucket)
}

func (t *AppendObjectCreatorTest) call() (o *gcs.Object, err error) {
	o, err = t.creator.Create(
		t.ctx,
		&t.srcObject,
		strings.NewReader(t.srcContents))

	return
}

////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////

func (t *AppendObjectCreatorTest) CallsCreateObject() {
	t.srcContents = "taco"

	// CreateObject
	var req *gcs.CreateObjectRequest
	ExpectCall(t.bucket, "CreateObject")(Any(), Any()).
		WillOnce(DoAll(SaveArg(1, &req), Return(nil, errors.New(""))))

	// Call
	t.call()

	AssertNe(nil, req)
	ExpectTrue(strings.HasPrefix(req.Name, prefix), "Name: %s", req.Name)

	b, err := ioutil.ReadAll(req.Contents)
	AssertEq(nil, err)
	ExpectEq(t.srcContents, string(b))
}

func (t *AppendObjectCreatorTest) CreateObjectFails() {
	AssertTrue(false, "TODO")
}

func (t *AppendObjectCreatorTest) CreateObjectReturnsPreconditionError() {
	AssertTrue(false, "TODO")
}

func (t *AppendObjectCreatorTest) CallsComposeObjects() {
	AssertTrue(false, "TODO")
}

func (t *AppendObjectCreatorTest) ComposeObjectsFails() {
	AssertTrue(false, "TODO")
}

func (t *AppendObjectCreatorTest) ComposeObjectsReturnsPreconditionError() {
	AssertTrue(false, "TODO")
}

func (t *AppendObjectCreatorTest) CallsDeleteObject() {
	AssertTrue(false, "TODO")
}

func (t *AppendObjectCreatorTest) DeleteObjectFails() {
	AssertTrue(false, "TODO")
}

func (t *AppendObjectCreatorTest) DeleteObjectSucceeds() {
	AssertTrue(false, "TODO")
}
