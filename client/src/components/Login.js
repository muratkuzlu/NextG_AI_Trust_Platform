import React, { useState } from "react";
import Form from "react-bootstrap/Form";
import { Button, Container, Row, Col } from "react-bootstrap";
import "./Login.css";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  function validateForm() {
    return email.length > 0 && password.length > 0;
  }

  let navigate = useNavigate();

  var reDrectPage = () => {
    navigate("/register");
    // localStorage.setItem("token", "logedOut");
  };

  function handleSubmit(event) {
    event.preventDefault();

    var details = {
      username: email,
      password: password,
      // 'grant_type': 'password'
    };

    var formBody = [];
    for (var property in details) {
      var encodedKey = encodeURIComponent(property);
      var encodedValue = encodeURIComponent(details[property]);

      formBody.push(encodedKey + "=" + encodedValue);
    }
    formBody = formBody.join("&");

    fetch("http://localhost:8000/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: formBody,
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("Result", data);
        // navigate('/home')
        if (!data.access_token) {
          setErrorMessage("Invalid credential");
          // console.log('not good cred')
        } else {
          // console.log('good credential')
          setErrorMessage("");
          // navigate('/home')
          navigate("/loaddata");
          localStorage.setItem("token", data.access_token);
        }
      });
  }

  return (
    <Row>
      <Col>
        <div className="Login">
          <Form onSubmit={handleSubmit}>
            <h3>Sign In</h3>
            <Form.Group size="lg" controlId="email">
              <Form.Label>Email</Form.Label>
              <Form.Control
                autoFocus
                // type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </Form.Group>
            <br></br>
            <Form.Group size="lg" controlId="password">
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </Form.Group>
            <br></br>
            <Button variant="primary" block type="submit">
              Login
            </Button>
            <Button
              className="float-end"
              variant="outline-secondary"
              onClick={reDrectPage}
            >
              Sign up
            </Button>
          </Form>
        </div>
      </Col>
    </Row>
  );
}
