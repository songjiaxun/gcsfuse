package main

import (
	"testing"

	. "github.com/jacobsa/ogletest"
)

const TestBucketName string = "gcsfuse-default-bucket"

func Test_Main(t *testing.T) { RunTests(t) }

////////////////////////////////////////////////////////////////////////
// Boilerplate
////////////////////////////////////////////////////////////////////////

type MainTest struct {
}

func init() { RegisterTestSuite(&MainTest{}) }

func (t *MainTest) TestCreateStorageHandleEnableStorageClientLibraryIsTrue() {
	storageHandle, err := createStorageHandle(&flagStorage{
		EnableStorageClientLibrary: true,
	})

	ExpectNe(nil, storageHandle)
	ExpectEq(nil, err)
}
