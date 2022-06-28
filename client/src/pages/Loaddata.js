import React from "react";
import { Button, Container, Row, Col } from "react-bootstrap";
import { useNavigate } from "react-router-dom";

const localHost_1 = "http://localhost:8504";
// {iframe_src}
function Loaddata() {
  let navigate = useNavigate();
  var reDrectPage = () => {
    navigate("/");
    localStorage.setItem("token", "logedOut");
  };

  return (
    <div>
      <Container fluid>
        <Row>
          <Col>
            <h4>Load Data</h4>
          </Col>
          <Col>
            <Button
              className="float-end"
              variant="outline-danger"
              onClick={reDrectPage}
            >
              Log Out
            </Button>
          </Col>
        </Row>
        <Row>
          <Col>
            <iframe
              title="labeler"
              src="http://localhost:8501"
              name="labelFrame"
              height="1000"
              // width="1500"
              width="100%"
            ></iframe>
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default Loaddata;
