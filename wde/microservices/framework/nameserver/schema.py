from pydantic import BaseModel


class ServerInfo(BaseModel):
    name: str
    host: str
    port: str | int
    protocol: str


class GetServicesRequest(BaseModel):
    name: str
    protocol: str


class GetServiceNamesRequest(BaseModel):
    protocol: str
